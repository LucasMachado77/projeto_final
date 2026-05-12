from __future__ import annotations

"""
Etapa 06b — Grad-CAM estendido (heatmap + overlay, acertos/erros).

Mantém a Etapa 06 (`step_06_gradcam.py`) no formato simples para comparação
baseline; use este script para curadoria clínica e exportação separada do heatmap.
"""

import argparse
import json
from pathlib import Path

from matplotlib import colormaps
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    # Define parâmetros de entrada para gerar visualizações Grad-CAM voltadas à leitura clínica.
    parser = argparse.ArgumentParser(
        description='Etapa 06b: Grad-CAM com overlay+heatmap e seleção por acertos/erros.',
    )
    parser.add_argument(
        '--split-csv',
        type=Path,
        default=Path('reports_adni/split_assignments_grouped_binary.csv'),
        help='CSV com colunas filepath, label e split.',
    )
    parser.add_argument(
        '--model-path',
        type=Path,
        default=Path('reports_adni/cnn_transfer_learning_e5_grouped_binary/best_model_resnet18.pth'),
        help='Caminho do modelo treinado (state_dict).',
    )
    parser.add_argument(
        '--image-size',
        type=int,
        default=224,
        help='Tamanho da imagem usado no treino da CNN.',
    )
    parser.add_argument(
        '--samples-per-class',
        type=int,
        default=2,
        help='No modo random_per_class: quantidade de exemplos por classe.',
    )
    parser.add_argument(
        '--selection-strategy',
        type=str,
        default='random_per_class',
        choices=['random_per_class', 'errors_and_correct'],
        help=(
            'random_per_class: amostragem equilibrada simples. '
            'errors_and_correct: erros + acertos no teste para contraste clínico.'
        ),
    )
    parser.add_argument(
        '--errors-per-class',
        type=int,
        default=2,
        help='No modo errors_and_correct: máximo de erros por rótulo verdadeiro.',
    )
    parser.add_argument(
        '--correct-per-class',
        type=int,
        default=2,
        help='No modo errors_and_correct: máximo de acertos por rótulo verdadeiro.',
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Tamanho do batch ao varrer o conjunto de teste (modo errors_and_correct).',
    )
    parser.add_argument(
        '--gradcam-target',
        type=str,
        default='predicted',
        choices=['predicted', 'true'],
        help='Classe no backward do Grad-CAM: predita ou verdadeira.',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Semente para seleção reprodutível de amostras.',
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('reports_adni/gradcam_grouped_binary_b'),
        help='Pasta de saída (separada da Etapa 06 para não sobrescrever).',
    )
    return parser.parse_args()


class SplitImageDataset(Dataset):
    # Dataset enxuto para leitura de imagens do CSV de teste com transformações determinísticas.
    def __init__(self, paths: list[Path], transform: transforms.Compose) -> None:
        self.paths = paths
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, Path]:
        image_path = self.paths[index]
        with Image.open(image_path) as image:
            rgb_image = image.convert('RGB')
            tensor = self.transform(rgb_image)
        return tensor, image_path


def load_split_dataframe(split_csv: Path) -> pd.DataFrame:
    # Carrega e valida estrutura mínima do CSV de split.
    if not split_csv.exists():
        raise FileNotFoundError(f'CSV não encontrado: {split_csv}')
    split_df = pd.read_csv(split_csv)
    required_columns = {'filepath', 'label', 'split'}
    missing_columns = required_columns.difference(split_df.columns)
    if missing_columns:
        raise ValueError(f'Colunas ausentes no CSV: {sorted(missing_columns)}')
    return split_df


def build_model(num_classes: int, model_path: Path, device: torch.device) -> nn.Module:
    # Reconstrói a ResNet18 com a mesma cabeça do treino e carrega os pesos.
    model = models.resnet18(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    # Compatibilidade: weights_only existe em PyTorch recente; fallback sem o argumento.
    try:
        state_dict = torch.load(model_path, map_location=device, weights_only=True)
    except TypeError:
        state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def get_eval_transform(image_size: int) -> transforms.Compose:
    # Mantém o mesmo pré-processamento de avaliação usado no treino.
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ],
    )


def normalize_heatmap(heatmap: np.ndarray) -> np.ndarray:
    # Normaliza para [0, 1] com segurança numérica.
    min_value = float(heatmap.min())
    max_value = float(heatmap.max())
    denom = max(max_value - min_value, 1e-8)
    return (heatmap - min_value) / denom


def build_overlay_image(original_rgb: np.ndarray, heatmap: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    # Converte heatmap em cor (jet) e mistura com imagem original para visualização.
    heatmap_colored = colormaps['jet'](heatmap)[:, :, :3]
    heatmap_uint8 = (heatmap_colored * 255.0).astype(np.uint8)
    overlay = (1.0 - alpha) * original_rgb.astype(np.float32) + alpha * heatmap_uint8.astype(np.float32)
    return np.clip(overlay, 0, 255).astype(np.uint8)


def build_jet_rgb_uint8(heatmap: np.ndarray) -> np.ndarray:
    # Heatmap colorido puro (sem overlay) — útil para apresentação médica em slides.
    heatmap_colored = colormaps['jet'](heatmap)[:, :, :3]
    return (heatmap_colored * 255.0).astype(np.uint8)


def compute_gradcam(
    model: nn.Module,
    image_tensor: torch.Tensor,
    target_layer: nn.Module,
    class_index: int,
) -> np.ndarray:
    # Calcula Grad-CAM com hooks na última camada convolucional.
    activations: list[torch.Tensor] = []
    gradients: list[torch.Tensor] = []

    def forward_hook(_module: nn.Module, _input: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        # Guarda ativações para cálculo posterior do mapa.
        activations.append(output.detach())

    def backward_hook(
        _module: nn.Module,
        grad_input: tuple[torch.Tensor, ...],
        grad_output: tuple[torch.Tensor, ...],
    ) -> None:
        # Guarda gradientes da saída da camada alvo.
        del grad_input
        gradients.append(grad_output[0].detach())

    forward_handle = target_layer.register_forward_hook(forward_hook)
    backward_handle = target_layer.register_full_backward_hook(backward_hook)

    try:
        model.zero_grad(set_to_none=True)
        logits = model(image_tensor)
        target_score = logits[:, class_index].sum()
        target_score.backward()

        activation_map = activations[0][0]
        gradient_map = gradients[0][0]
        weights = gradient_map.mean(dim=(1, 2), keepdim=True)
        cam = (weights * activation_map).sum(dim=0)
        cam = torch.relu(cam)
        cam_np = cam.cpu().numpy()
        return normalize_heatmap(cam_np)
    finally:
        # Remove hooks para evitar acúmulo entre iterações.
        forward_handle.remove()
        backward_handle.remove()


def build_progress_bar(dataloader: DataLoader, description: str):
    # Mantém barra estável no PowerShell evitando quebra de linha por largura dinâmica.
    return tqdm(
        dataloader,
        desc=description,
        leave=False,
        dynamic_ncols=False,
        ncols=100,
        mininterval=1.0,
        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]',
    )


def collect_test_predictions(
    model: nn.Module,
    test_df: pd.DataFrame,
    eval_transform: transforms.Compose,
    label_to_id: dict[str, int],
    device: torch.device,
    batch_size: int,
) -> pd.DataFrame:
    # Varre o teste em batches e devolve predições alinhadas ao caminho da imagem.
    existing_paths: list[Path] = []
    labels: list[str] = []
    for row in test_df.itertuples(index=False):
        path = Path(str(row.filepath))
        if path.exists():
            existing_paths.append(path)
            labels.append(str(row.label))

    if not existing_paths:
        raise ValueError('Nenhum filepath válido no split de teste.')

    dataset = SplitImageDataset(existing_paths, eval_transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    predictions: list[int] = []
    with torch.no_grad():
        for batch_tensors, _ in build_progress_bar(dataloader=loader, description='Inferência no teste'):
            batch_tensors = batch_tensors.to(device)
            logits = model(batch_tensors)
            batch_pred = torch.argmax(logits, dim=1).detach().cpu().numpy().tolist()
            predictions.extend(int(value) for value in batch_pred)

    id_to_label = {idx: label for label, idx in label_to_id.items()}
    pred_labels = [id_to_label[idx] for idx in predictions]

    results_df = pd.DataFrame(
        {
            'filepath': [str(path) for path in existing_paths],
            'true_label': labels,
            'predicted_label': pred_labels,
        },
    )
    results_df['is_correct'] = results_df['true_label'] == results_df['predicted_label']
    return results_df


def sample_balanced_errors_and_correct(
    results_df: pd.DataFrame,
    classes: list[str],
    errors_per_class: int,
    correct_per_class: int,
    seed: int,
) -> pd.DataFrame:
    # Seleciona erros e acertos balanceados por classe verdadeira para narrativa clínica.
    rng = np.random.default_rng(seed)
    selected_frames: list[pd.DataFrame] = []

    for class_name in classes:
        class_results = results_df[results_df['true_label'] == class_name]
        wrong = class_results[~class_results['is_correct']]
        right = class_results[class_results['is_correct']]

        wrong_sample = wrong.sample(n=min(errors_per_class, len(wrong)), random_state=int(rng.integers(0, 10_000_000)))
        right_sample = right.sample(n=min(correct_per_class, len(right)), random_state=int(rng.integers(0, 10_000_000)))
        selected_frames.append(wrong_sample)
        selected_frames.append(right_sample)

    selected_df = pd.concat(selected_frames, axis=0).drop_duplicates(subset=['filepath']).reset_index(drop=True)
    return selected_df


def sample_random_per_class(test_df: pd.DataFrame, classes: list[str], samples_per_class: int, seed: int) -> pd.DataFrame:
    # Seleção clássica: amostras aleatórias por classe a partir do CSV de teste.
    frames: list[pd.DataFrame] = []
    for class_name in classes:
        class_df = test_df[test_df['label'] == class_name].copy()
        count = min(samples_per_class, len(class_df))
        frames.append(class_df.sample(n=count, random_state=seed))
    return pd.concat(frames, axis=0).reset_index(drop=True)


def main() -> None:
    # Orquestra inferência opcional no teste, seleção de casos e geração de Grad-CAM.
    args = parse_args()
    np.random.seed(args.seed)

    split_df = load_split_dataframe(args.split_csv)
    test_df = split_df[split_df['split'] == 'test'].copy()
    if test_df.empty:
        raise ValueError('Split de teste vazio no CSV informado.')

    classes = sorted(test_df['label'].unique().tolist())
    class_to_index = {class_name: idx for idx, class_name in enumerate(classes)}
    index_to_class = {idx: class_name for class_name, idx in class_to_index.items()}

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if not args.model_path.exists():
        raise FileNotFoundError(f'Modelo não encontrado: {args.model_path}')
    model = build_model(
        num_classes=len(classes),
        model_path=args.model_path,
        device=device,
    )

    target_layer = model.layer4[1].conv2
    eval_transform = get_eval_transform(args.image_size)

    if args.selection_strategy == 'random_per_class':
        selected_df = sample_random_per_class(
            test_df=test_df,
            classes=classes,
            samples_per_class=args.samples_per_class,
            seed=args.seed,
        )
        selected_df = selected_df.rename(columns={'label': 'true_label'})
    else:
        results_df = collect_test_predictions(
            model=model,
            test_df=test_df,
            eval_transform=eval_transform,
            label_to_id=class_to_index,
            device=device,
            batch_size=args.batch_size,
        )
        selected_df = sample_balanced_errors_and_correct(
            results_df=results_df,
            classes=classes,
            errors_per_class=args.errors_per_class,
            correct_per_class=args.correct_per_class,
            seed=args.seed,
        )

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata_rows: list[dict[str, object]] = []
    for row_idx, row in enumerate(selected_df.itertuples(index=False), start=1):
        image_path = Path(str(row.filepath))
        if not image_path.exists():
            continue

        true_label = str(row.true_label)

        with Image.open(image_path) as image:
            rgb_image = image.convert('RGB')
            resized_rgb = rgb_image.resize((args.image_size, args.image_size))
            original_np = np.array(resized_rgb, dtype=np.uint8)

        image_tensor = eval_transform(resized_rgb).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(image_tensor)
            predicted_index = int(torch.argmax(logits, dim=1).item())
            predicted_label = index_to_class[predicted_index]

        if args.selection_strategy == 'random_per_class':
            is_correct = true_label == predicted_label
        else:
            is_correct = bool(getattr(row, 'is_correct'))

        if args.gradcam_target == 'true':
            cam_class_index = class_to_index[true_label]
            cam_class_label = true_label
        else:
            cam_class_index = predicted_index
            cam_class_label = predicted_label

        cam = compute_gradcam(
            model=model,
            image_tensor=image_tensor,
            target_layer=target_layer,
            class_index=cam_class_index,
        )

        cam_image = Image.fromarray((cam * 255.0).astype(np.uint8))
        cam_resized = cam_image.resize((args.image_size, args.image_size), resample=Image.BILINEAR)
        cam_resized_np = np.array(cam_resized, dtype=np.float32) / 255.0

        overlay_np = build_overlay_image(original_np, cam_resized_np, alpha=0.45)
        heatmap_rgb = build_jet_rgb_uint8(cam_resized_np)

        safe_true = true_label.replace(' ', '_')
        safe_pred = predicted_label.replace(' ', '_')
        safe_cam = cam_class_label.replace(' ', '_')
        base_name = f'{safe_true}_case{row_idx:02d}_pred_{safe_pred}_camcls_{safe_cam}'
        overlay_path = output_dir / f'{base_name}_overlay.png'
        heatmap_path = output_dir / f'{base_name}_heatmap.png'

        Image.fromarray(overlay_np).save(overlay_path)
        Image.fromarray(heatmap_rgb).save(heatmap_path)

        metadata_rows.append(
            {
                'image_path': str(image_path),
                'true_label': true_label,
                'predicted_label': predicted_label,
                'is_correct': is_correct,
                'gradcam_target': args.gradcam_target,
                'gradcam_class_for_backward': cam_class_label,
                'overlay_file': str(overlay_path),
                'heatmap_file': str(heatmap_path),
            },
        )

    metadata_df = pd.DataFrame(metadata_rows)
    metadata_csv = output_dir / 'gradcam_metadata.csv'
    metadata_df.to_csv(metadata_csv, index=False)

    class_mapping_path = output_dir / 'class_mapping.json'
    with class_mapping_path.open('w', encoding='utf-8') as mapping_file:
        json.dump(class_to_index, mapping_file, indent=2, ensure_ascii=False)

    print('\nGrad-CAM (06b) concluído:')
    print(f'- Pasta de imagens: {output_dir.resolve()}')
    print(f'- Metadados: {metadata_csv.resolve()}')
    print(f'- Mapeamento de classes: {class_mapping_path.resolve()}')
    print(f'- Total de visualizações geradas: {len(metadata_df)}')


if __name__ == '__main__':
    # Mantém execução direta do script via linha de comando.
    main()
