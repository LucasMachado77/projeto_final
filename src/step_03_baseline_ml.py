from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    # Define argumentos para controlar caminho de entrada e hiperparâmetros do baseline.
    parser = argparse.ArgumentParser(
        description='Treina baseline clássico com LogisticRegression usando o CSV de splits.',
    )
    parser.add_argument(
        '--split-csv',
        type=Path,
        default=Path('reports/split_assignments.csv'),
        help='Arquivo CSV gerado no passo 02.',
    )
    parser.add_argument(
        '--image-size',
        type=int,
        default=32,
        help='Tamanho da imagem quadrada para extração de features simples.',
    )
    parser.add_argument(
        '--max-iter',
        type=int,
        default=300,
        help='Número máximo de iterações do LogisticRegression.',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Semente para garantir reprodutibilidade.',
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('reports/baseline_ml'),
        help='Pasta para salvar métricas e matriz de confusão.',
    )
    return parser.parse_args()


def load_split_dataframe(split_csv: Path) -> pd.DataFrame:
    # Lê o CSV de splits e valida colunas essenciais para o pipeline.
    if not split_csv.exists():
        raise FileNotFoundError(f'Arquivo de split não encontrado: {split_csv}')

    split_df = pd.read_csv(split_csv)
    required_columns = {'filepath', 'label', 'split'}
    missing_columns = required_columns.difference(split_df.columns)
    if missing_columns:
        raise ValueError(f'Colunas ausentes no CSV: {sorted(missing_columns)}')

    return split_df


def load_single_image(image_path: Path, image_size: int) -> np.ndarray:
    # Abre, converte para escala de cinza e redimensiona para vetor fixo de features.
    with Image.open(image_path) as image:
        gray_image = image.convert('L')
        resized_image = gray_image.resize((image_size, image_size))
        image_array = np.asarray(resized_image, dtype=np.float32) / 255.0
        return image_array.flatten()


def build_feature_matrix(
    split_df: pd.DataFrame,
    image_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    # Constrói matriz X e vetor y com leitura progressiva das imagens.
    feature_list: list[np.ndarray] = []
    label_list: list[str] = []

    records = split_df[['filepath', 'label']].to_dict(orient='records')
    for record in tqdm(records, desc='Extraindo features'):
        image_path = Path(record['filepath'])
        label = str(record['label'])

        # Ignora imagens ausentes para evitar quebra total do processamento.
        if not image_path.exists():
            continue

        feature_vector = load_single_image(image_path=image_path, image_size=image_size)
        feature_list.append(feature_vector)
        label_list.append(label)

    if not feature_list:
        raise ValueError('Nenhuma imagem válida foi carregada para treino/avaliação.')

    x_matrix = np.vstack(feature_list)
    y_vector = np.array(label_list)
    return x_matrix, y_vector


def split_dataframe_by_set(split_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # Separa DataFrames de treino, validação e teste para fluxo mais claro.
    train_df = split_df[split_df['split'] == 'train'].copy()
    val_df = split_df[split_df['split'] == 'val'].copy()
    test_df = split_df[split_df['split'] == 'test'].copy()

    if train_df.empty or val_df.empty or test_df.empty:
        raise ValueError('Um dos splits está vazio; revise o CSV do passo 02.')

    return train_df, val_df, test_df


def train_and_evaluate(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    image_size: int,
    max_iter: int,
    seed: int,
) -> tuple[dict[str, float], pd.DataFrame, dict[str, object]]:
    # Extrai features de cada split e codifica labels em IDs numéricos.
    x_train, y_train_text = build_feature_matrix(split_df=train_df, image_size=image_size)
    x_val, y_val_text = build_feature_matrix(split_df=val_df, image_size=image_size)
    x_test, y_test_text = build_feature_matrix(split_df=test_df, image_size=image_size)

    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train_text)
    y_val = label_encoder.transform(y_val_text)
    y_test = label_encoder.transform(y_test_text)

    # Treina baseline com balanceamento de classes para reduzir viés.
    model = LogisticRegression(
        max_iter=max_iter,
        random_state=seed,
        class_weight='balanced',
        solver='saga',
    )
    model.fit(x_train, y_train)

    # Avalia o desempenho em validação e teste.
    y_val_pred = model.predict(x_val)
    y_test_pred = model.predict(x_test)
    val_accuracy = accuracy_score(y_val, y_val_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)

    # Gera relatório detalhado por classe para análise posterior.
    test_report = classification_report(
        y_test,
        y_test_pred,
        target_names=label_encoder.classes_,
        output_dict=True,
        zero_division=0,
    )
    confusion = confusion_matrix(y_test, y_test_pred)
    confusion_df = pd.DataFrame(
        confusion,
        index=label_encoder.classes_,
        columns=label_encoder.classes_,
    )

    metrics = {
        'val_accuracy': float(val_accuracy),
        'test_accuracy': float(test_accuracy),
        'train_samples': int(len(x_train)),
        'val_samples': int(len(x_val)),
        'test_samples': int(len(x_test)),
    }
    metadata = {
        'label_mapping': {
            str(class_name): int(class_id)
            for class_id, class_name in enumerate(label_encoder.classes_)
        },
        'classification_report_test': test_report,
    }
    return metrics, confusion_df, metadata


def save_outputs(
    output_dir: Path,
    metrics: dict[str, float],
    confusion_df: pd.DataFrame,
    metadata: dict[str, object],
) -> None:
    # Salva artefatos em disco para uso no relatório e comparação futura.
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = output_dir / 'metrics.json'
    confusion_path = output_dir / 'confusion_matrix.csv'
    report_path = output_dir / 'classification_report_test.json'

    with metrics_path.open('w', encoding='utf-8') as metrics_file:
        json.dump(metrics, metrics_file, indent=2, ensure_ascii=False)

    confusion_df.to_csv(confusion_path, index=True)

    with report_path.open('w', encoding='utf-8') as report_file:
        json.dump(metadata['classification_report_test'], report_file, indent=2, ensure_ascii=False)

    print('\nArquivos gerados:')
    print(f'- {metrics_path.resolve()}')
    print(f'- {confusion_path.resolve()}')
    print(f'- {report_path.resolve()}')


def main() -> None:
    # Coordena leitura de dados, treino do baseline e persistência de resultados.
    args = parse_args()
    split_df = load_split_dataframe(split_csv=args.split_csv)
    train_df, val_df, test_df = split_dataframe_by_set(split_df=split_df)

    metrics, confusion_df, metadata = train_and_evaluate(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        image_size=args.image_size,
        max_iter=args.max_iter,
        seed=args.seed,
    )

    save_outputs(
        output_dir=args.output_dir,
        metrics=metrics,
        confusion_df=confusion_df,
        metadata=metadata,
    )

    print('\nMétricas principais:')
    print(f"- Val Accuracy: {metrics['val_accuracy']:.4f}")
    print(f"- Test Accuracy: {metrics['test_accuracy']:.4f}")


if __name__ == '__main__':
    # Mantém o padrão de execução direta por CLI.
    main()
