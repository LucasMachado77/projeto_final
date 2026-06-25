from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


DEFAULT_MODEL_NAME = 'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Extrai embeddings BiomedCLIP e treina um linear probe no ADNI.',
    )
    parser.add_argument(
        '--split-csv',
        type=Path,
        default=Path('reports_adni/split_assignments_grouped_binary.csv'),
        help='CSV com colunas filepath, label e split.',
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('reports_adni/biomedclip_embeddings_grouped_binary'),
        help='Pasta de saida para metricas e artefatos.',
    )
    parser.add_argument(
        '--model-name',
        default=DEFAULT_MODEL_NAME,
        help='Identificador do modelo BiomedCLIP no OpenCLIP/Hugging Face Hub.',
    )
    parser.add_argument('--batch-size', type=int, default=32, help='Tamanho do batch para extrair embeddings.')
    parser.add_argument('--num-workers', type=int, default=0, help='Workers do DataLoader.')
    parser.add_argument('--classifier-c', type=float, default=1.0, help='Parametro C da LogisticRegression.')
    parser.add_argument('--max-iter', type=int, default=1000, help='Maximo de iteracoes da LogisticRegression.')
    parser.add_argument('--seed', type=int, default=42, help='Semente para reproducibilidade.')
    parser.add_argument(
        '--device',
        choices=('auto', 'cpu', 'cuda'),
        default='auto',
        help='Dispositivo para extracao de embeddings.',
    )
    parser.add_argument(
        '--save-embeddings',
        action='store_true',
        help='Salva embeddings .npz por split. Por padrao salva apenas metricas e predicoes.',
    )
    parser.add_argument(
        '--skip-missing-paths',
        action='store_true',
        help='Remove linhas cujo filepath nao existe, em vez de falhar.',
    )
    parser.add_argument(
        '--path-prefix-from',
        type=Path,
        default=None,
        help='Prefixo antigo nos filepaths do CSV, caso os dados tenham sido movidos.',
    )
    parser.add_argument(
        '--path-prefix-to',
        type=Path,
        default=None,
        help='Novo prefixo para substituir --path-prefix-from.',
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def import_open_clip():
    try:
        import open_clip
    except ModuleNotFoundError as exc:
        if exc.name == 'open_clip':
            raise ModuleNotFoundError(
                'open_clip_torch nao esta instalado. Rode: pip install -r requirements.txt',
            ) from exc
        raise
    return open_clip


def choose_device(requested_device: str) -> torch.device:
    if requested_device == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if requested_device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA foi solicitada, mas torch.cuda.is_available() retornou False.')
    return torch.device(requested_device)


def remap_path(path: Path, prefix_from: Path | None, prefix_to: Path | None) -> Path:
    if prefix_from is None or prefix_to is None:
        return path

    raw_path = str(path)
    raw_from = str(prefix_from)
    if os.name == 'nt':
        path_cmp = raw_path.lower()
        from_cmp = raw_from.lower()
    else:
        path_cmp = raw_path
        from_cmp = raw_from

    if not path_cmp.startswith(from_cmp):
        return path

    suffix = raw_path[len(raw_from) :].lstrip('\\/')
    return prefix_to / suffix


def load_split_dataframe(
    split_csv: Path,
    skip_missing_paths: bool,
    path_prefix_from: Path | None,
    path_prefix_to: Path | None,
) -> pd.DataFrame:
    if not split_csv.exists():
        raise FileNotFoundError(f'CSV de split nao encontrado: {split_csv}')

    split_df = pd.read_csv(split_csv)
    required_columns = {'filepath', 'label', 'split'}
    missing_columns = required_columns.difference(split_df.columns)
    if missing_columns:
        raise ValueError(f'Colunas ausentes no CSV: {sorted(missing_columns)}')

    split_df = split_df.copy()
    split_df['filepath'] = split_df['filepath'].apply(
        lambda value: str(remap_path(Path(str(value)), path_prefix_from, path_prefix_to)),
    )
    split_df['label'] = split_df['label'].astype(str)
    split_df['split'] = split_df['split'].astype(str)
    split_df['path_exists'] = split_df['filepath'].apply(lambda value: Path(str(value)).is_file())

    missing_df = split_df[~split_df['path_exists']]
    if not missing_df.empty and not skip_missing_paths:
        examples = '\n'.join(str(path) for path in missing_df['filepath'].head(5))
        raise FileNotFoundError(
            f'{len(missing_df)} imagens do CSV nao existem no disco. Exemplos:\n{examples}',
        )
    if skip_missing_paths:
        split_df = split_df[split_df['path_exists']].copy()

    return split_df.drop(columns=['path_exists'])


def split_dataframe_by_set(split_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_df = split_df[split_df['split'] == 'train'].copy()
    val_df = split_df[split_df['split'] == 'val'].copy()
    test_df = split_df[split_df['split'] == 'test'].copy()
    if train_df.empty or val_df.empty or test_df.empty:
        raise ValueError('Um dos splits esta vazio; revise o CSV de entrada.')
    return train_df, val_df, test_df


class ImageEmbeddingDataset(Dataset):
    def __init__(self, split_df: pd.DataFrame, preprocess: Any) -> None:
        self.split_df = split_df.reset_index(drop=True).copy()
        self.preprocess = preprocess

    def __len__(self) -> int:
        return len(self.split_df)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, str, str]:
        row = self.split_df.iloc[index]
        image_path = Path(str(row['filepath']))
        label_id = int(row['label_id'])
        label_name = str(row['label'])

        with Image.open(image_path) as image:
            rgb_image = image.convert('RGB')
            image_tensor = self.preprocess(rgb_image)

        return image_tensor, label_id, str(image_path), label_name


def load_biomedclip_model(model_name: str, device: torch.device):
    open_clip = import_open_clip()
    try:
        model, preprocess = open_clip.create_model_from_pretrained(model_name)
    except Exception as exc:
        raise RuntimeError(
            'Nao foi possivel carregar o BiomedCLIP. Na primeira execucao, '
            'o OpenCLIP precisa baixar pesos do Hugging Face Hub.',
        ) from exc

    model.to(device)
    model.eval()
    return model, preprocess


def extract_embeddings(
    model: torch.nn.Module,
    preprocess: Any,
    split_df: pd.DataFrame,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    description: str,
) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    dataset = ImageEmbeddingDataset(split_df=split_df, preprocess=preprocess)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    embedding_batches: list[np.ndarray] = []
    label_batches: list[np.ndarray] = []
    filepaths: list[str] = []
    label_names: list[str] = []

    with torch.no_grad():
        for images, labels, paths, names in tqdm(loader, desc=description):
            images = images.to(device)
            features = model.encode_image(images)
            if isinstance(features, tuple):
                features = features[0]
            features = features.float()
            features = features / torch.clamp(features.norm(dim=-1, keepdim=True), min=1e-12)

            embedding_batches.append(features.detach().cpu().numpy())
            label_batches.append(labels.detach().cpu().numpy())
            filepaths.extend(list(paths))
            label_names.extend(list(names))

    if not embedding_batches:
        raise ValueError(f'Nenhuma imagem valida processada em {description}.')

    embeddings = np.vstack(embedding_batches).astype(np.float32)
    labels_array = np.concatenate(label_batches).astype(np.int64)
    return embeddings, labels_array, filepaths, label_names


def build_classifier(seed: int, classifier_c: float, max_iter: int) -> Pipeline:
    return Pipeline(
        steps=[
            ('scaler', StandardScaler()),
            (
                'classifier',
                LogisticRegression(
                    C=classifier_c,
                    class_weight='balanced',
                    max_iter=max_iter,
                    random_state=seed,
                    solver='lbfgs',
                ),
            ),
        ],
    )


def predict_with_probabilities(
    classifier: Pipeline,
    embeddings: np.ndarray,
    num_classes: int,
) -> tuple[np.ndarray, np.ndarray]:
    predictions = classifier.predict(embeddings)
    classifier_step: LogisticRegression = classifier.named_steps['classifier']
    raw_probabilities = classifier.predict_proba(embeddings)
    probabilities = np.zeros((len(embeddings), num_classes), dtype=np.float64)
    for probability_column, class_id in enumerate(classifier_step.classes_):
        probabilities[:, int(class_id)] = raw_probabilities[:, probability_column]
    return predictions.astype(np.int64), probabilities


def compute_log_loss(labels: np.ndarray, probabilities: np.ndarray, num_classes: int) -> float:
    return float(log_loss(labels, probabilities, labels=list(range(num_classes))))


def make_confusion_df(
    labels: np.ndarray,
    predictions: np.ndarray,
    target_names: list[str],
) -> pd.DataFrame:
    class_ids = list(range(len(target_names)))
    confusion = confusion_matrix(labels, predictions, labels=class_ids)
    return pd.DataFrame(confusion, index=target_names, columns=target_names)


def build_predictions_df(
    filepaths: list[str],
    labels: np.ndarray,
    predictions: np.ndarray,
    probabilities: np.ndarray,
    target_names: list[str],
) -> pd.DataFrame:
    rows = {
        'filepath': filepaths,
        'true_label': [target_names[int(label_id)] for label_id in labels],
        'pred_label': [target_names[int(label_id)] for label_id in predictions],
        'correct': labels == predictions,
    }
    predictions_df = pd.DataFrame(rows)
    for class_id, class_name in enumerate(target_names):
        safe_name = class_name.replace(' ', '_').replace('/', '_')
        predictions_df[f'prob_{safe_name}'] = probabilities[:, class_id]
    return predictions_df


def save_embeddings_npz(
    output_dir: Path,
    split_name: str,
    embeddings: np.ndarray,
    labels: np.ndarray,
    filepaths: list[str],
    label_names: list[str],
    target_names: list[str],
) -> None:
    np.savez_compressed(
        output_dir / f'embeddings_{split_name}.npz',
        embeddings=embeddings,
        labels=labels,
        filepaths=np.array(filepaths),
        label_names=np.array(label_names),
        classes=np.array(target_names),
    )


def save_outputs(
    output_dir: Path,
    metrics: dict[str, object],
    confusion_df: pd.DataFrame,
    classification_report_dict: dict[str, object],
    label_mapping: dict[str, int],
    val_predictions_df: pd.DataFrame,
    test_predictions_df: pd.DataFrame,
    embedding_summary: dict[str, object],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'metrics.json').write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    (output_dir / 'classification_report_test.json').write_text(
        json.dumps(classification_report_dict, indent=2, ensure_ascii=False) + '\n',
        encoding='utf-8',
    )
    (output_dir / 'class_mapping.json').write_text(
        json.dumps(label_mapping, indent=2, ensure_ascii=False) + '\n',
        encoding='utf-8',
    )
    (output_dir / 'embedding_summary.json').write_text(
        json.dumps(embedding_summary, indent=2, ensure_ascii=False) + '\n',
        encoding='utf-8',
    )
    confusion_df.to_csv(output_dir / 'confusion_matrix.csv', index=True)
    val_predictions_df.to_csv(output_dir / 'predictions_val.csv', index=False)
    test_predictions_df.to_csv(output_dir / 'predictions_test.csv', index=False)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    split_df = load_split_dataframe(
        split_csv=args.split_csv,
        skip_missing_paths=args.skip_missing_paths,
        path_prefix_from=args.path_prefix_from,
        path_prefix_to=args.path_prefix_to,
    )

    label_encoder = LabelEncoder()
    split_df['label_id'] = label_encoder.fit_transform(split_df['label'])
    target_names = [str(class_name) for class_name in label_encoder.classes_]
    label_mapping = {class_name: int(class_id) for class_id, class_name in enumerate(target_names)}
    train_df, val_df, test_df = split_dataframe_by_set(split_df)

    device = choose_device(args.device)
    print(f'Usando dispositivo: {device}')
    print(f'Carregando modelo: {args.model_name}')
    model, preprocess = load_biomedclip_model(model_name=args.model_name, device=device)

    x_train, y_train, train_paths, train_names = extract_embeddings(
        model=model,
        preprocess=preprocess,
        split_df=train_df,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
        description='Embeddings train',
    )
    x_val, y_val, val_paths, val_names = extract_embeddings(
        model=model,
        preprocess=preprocess,
        split_df=val_df,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
        description='Embeddings val',
    )
    x_test, y_test, test_paths, test_names = extract_embeddings(
        model=model,
        preprocess=preprocess,
        split_df=test_df,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
        description='Embeddings test',
    )

    classifier = build_classifier(seed=args.seed, classifier_c=args.classifier_c, max_iter=args.max_iter)
    classifier.fit(x_train, y_train)

    val_predictions, val_probabilities = predict_with_probabilities(classifier, x_val, len(target_names))
    test_predictions, test_probabilities = predict_with_probabilities(classifier, x_test, len(target_names))
    val_accuracy = float(accuracy_score(y_val, val_predictions))
    test_accuracy = float(accuracy_score(y_test, test_predictions))
    val_loss = compute_log_loss(y_val, val_probabilities, len(target_names))
    test_loss = compute_log_loss(y_test, test_probabilities, len(target_names))

    confusion_df = make_confusion_df(y_test, test_predictions, target_names)
    classification_report_dict = classification_report(
        y_test,
        test_predictions,
        labels=list(range(len(target_names))),
        target_names=target_names,
        output_dict=True,
        zero_division=0,
    )
    val_predictions_df = build_predictions_df(val_paths, y_val, val_predictions, val_probabilities, target_names)
    test_predictions_df = build_predictions_df(test_paths, y_test, test_predictions, test_probabilities, target_names)

    embedding_summary = {
        'embedding_dim': int(x_train.shape[1]),
        'embedding_normalized_l2': True,
        'train_shape': list(x_train.shape),
        'val_shape': list(x_val.shape),
        'test_shape': list(x_test.shape),
        'save_embeddings': bool(args.save_embeddings),
    }
    metrics = {
        'framework': 'OpenCLIP',
        'model': 'BiomedCLIP',
        'model_name': args.model_name,
        'pretrained': True,
        'task': 'image_embedding_linear_probe',
        'classifier': 'StandardScaler + LogisticRegression',
        'class_weight': 'balanced',
        'classifier_c': float(args.classifier_c),
        'max_iter': int(args.max_iter),
        'seed': int(args.seed),
        'device': str(device),
        'batch_size': int(args.batch_size),
        'train_samples': int(len(y_train)),
        'val_samples': int(len(y_val)),
        'test_samples': int(len(y_test)),
        'embedding_dim': int(x_train.shape[1]),
        'val_loss': val_loss,
        'val_accuracy': val_accuracy,
        'test_loss': test_loss,
        'test_accuracy': test_accuracy,
    }

    save_outputs(
        output_dir=args.output_dir,
        metrics=metrics,
        confusion_df=confusion_df,
        classification_report_dict=classification_report_dict,
        label_mapping=label_mapping,
        val_predictions_df=val_predictions_df,
        test_predictions_df=test_predictions_df,
        embedding_summary=embedding_summary,
    )

    if args.save_embeddings:
        save_embeddings_npz(args.output_dir, 'train', x_train, y_train, train_paths, train_names, target_names)
        save_embeddings_npz(args.output_dir, 'val', x_val, y_val, val_paths, val_names, target_names)
        save_embeddings_npz(args.output_dir, 'test', x_test, y_test, test_paths, test_names, target_names)

    print('\nBiomedCLIP embeddings concluido:')
    print(f'- Pasta: {args.output_dir.resolve()}')
    print(f'- Embedding dim: {x_train.shape[1]}')
    print(f'- Val Accuracy: {val_accuracy:.4f}')
    print(f'- Test Accuracy: {test_accuracy:.4f}')


if __name__ == '__main__':
    main()
