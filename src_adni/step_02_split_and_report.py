from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


# Define extensões válidas para reduzir risco de ler arquivos não relacionados.
VALID_IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}


def parse_args() -> argparse.Namespace:
    # Lê parâmetros de execução para deixar o script reutilizável em diferentes cenários.
    parser = argparse.ArgumentParser(
        description=(
            'Gera split estratificado (train/val/test) por imagem (não por paciente). '
            'No ADNI use step_02b + step_02c para evitar vazamento entre splits.'
        ),
    )
    parser.add_argument(
        '--dataset-dir',
        type=Path,
        required=True,
        help='Pasta raiz com subpastas de classes.',
    )
    parser.add_argument(
        '--output-csv',
        type=Path,
        default=Path('reports_adni/split_assignments.csv'),
        help='Arquivo CSV de saída com caminho, classe e split.',
    )
    parser.add_argument(
        '--train-size',
        type=float,
        default=0.7,
        help='Proporção para treino.',
    )
    parser.add_argument(
        '--val-size',
        type=float,
        default=0.15,
        help='Proporção para validação.',
    )
    parser.add_argument(
        '--test-size',
        type=float,
        default=0.15,
        help='Proporção para teste.',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Semente para reprodutibilidade.',
    )
    return parser.parse_args()


def validate_sizes(train_size: float, val_size: float, test_size: float) -> None:
    # Garante que as proporções estão corretas antes de iniciar o processamento.
    total = train_size + val_size + test_size
    if abs(total - 1.0) > 1e-9:
        raise ValueError(
            f'As proporções devem somar 1.0, mas somaram {total:.6f}.',
        )


def collect_dataset_rows(dataset_dir: Path) -> pd.DataFrame:
    # Percorre cada classe e coleta apenas arquivos de imagem válidos.
    rows: list[dict[str, str]] = []
    class_dirs = [path for path in dataset_dir.iterdir() if path.is_dir()]
    class_dirs.sort(key=lambda path: path.name.lower())

    for class_dir in class_dirs:
        class_name = class_dir.name
        for file_path in class_dir.rglob('*'):
            if file_path.is_file() and file_path.suffix.lower() in VALID_IMAGE_EXTENSIONS:
                rows.append(
                    {
                        'filepath': str(file_path.resolve()),
                        'label': class_name,
                    },
                )

    if not rows:
        raise ValueError('Nenhuma imagem válida encontrada no dataset informado.')

    return pd.DataFrame(rows)


def build_stratified_split(
    dataset_df: pd.DataFrame,
    train_size: float,
    val_size: float,
    test_size: float,
    seed: int,
) -> pd.DataFrame:
    # Divide em treino e temporário mantendo distribuição de classes.
    train_df, temp_df = train_test_split(
        dataset_df,
        test_size=(1.0 - train_size),
        random_state=seed,
        stratify=dataset_df['label'],
    )

    # Divide temporário em validação e teste, também com estratificação.
    val_ratio_inside_temp = val_size / (val_size + test_size)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=(1.0 - val_ratio_inside_temp),
        random_state=seed,
        stratify=temp_df['label'],
    )

    # Marca cada subconjunto para facilitar uso em etapas futuras.
    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()
    train_df['split'] = 'train'
    val_df['split'] = 'val'
    test_df['split'] = 'test'

    # Concatena tudo em um único DataFrame de referência.
    final_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    return final_df


def print_report(split_df: pd.DataFrame) -> None:
    # Exibe resumo de quantidade e percentual por split e classe.
    print('\nResumo por split e classe:\n')
    counts = (
        split_df.groupby(['split', 'label'])
        .size()
        .reset_index(name='count')
        .sort_values(by=['split', 'label'])
    )
    print(counts.to_string(index=False))

    print('\nResumo geral por split:\n')
    split_counts = split_df['split'].value_counts().rename_axis('split').reset_index(name='count')
    split_counts = split_counts.sort_values(by='split')
    split_counts['percent'] = (split_counts['count'] / len(split_df)) * 100.0
    print(split_counts.to_string(index=False, formatters={'percent': '{:.2f}%'.format}))

    print(f'\nTotal de imagens: {len(split_df)}')


def main() -> None:
    # Orquestra validação, leitura, split e gravação dos resultados.
    args = parse_args()
    dataset_dir: Path = args.dataset_dir
    output_csv: Path = args.output_csv

    if not dataset_dir.exists():
        raise FileNotFoundError(f'O diretório não existe: {dataset_dir}')
    if not dataset_dir.is_dir():
        raise NotADirectoryError(f'O caminho informado não é uma pasta: {dataset_dir}')

    validate_sizes(args.train_size, args.val_size, args.test_size)
    dataset_df = collect_dataset_rows(dataset_dir)
    split_df = build_stratified_split(
        dataset_df=dataset_df,
        train_size=args.train_size,
        val_size=args.val_size,
        test_size=args.test_size,
        seed=args.seed,
    )

    # Garante criação da pasta de saída antes de salvar o CSV.
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    split_df.to_csv(output_csv, index=False)

    print_report(split_df)
    print(f'\nCSV gerado em: {output_csv.resolve()}')


if __name__ == '__main__':
    # Permite execução direta via linha de comando.
    main()
