from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


# Define extensões válidas para evitar arquivos não relacionados ao dataset.
VALID_IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
# Captura IDs: OASIS (OAS1_XXXX) ou ADNI (NNN_S_NNNN no nome do ficheiro).
OASIS_SUBJECT_PATTERN = re.compile(r'(OAS1_\d{4})', flags=re.IGNORECASE)
ADNI_SUBJECT_PATTERN = re.compile(r'(\d{3}_S_\d{4})')


def parse_args() -> argparse.Namespace:
    # Recebe parâmetros para controlar split e caminhos de entrada/saída.
    parser = argparse.ArgumentParser(
        description='Gera split estratificado por paciente (group split) e relatório.',
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
        default=Path('reports_adni/split_assignments_grouped.csv'),
        help='CSV de saída com filepath, label, subject_id e split.',
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
    # Garante que as proporções do split somam exatamente 1.
    total = train_size + val_size + test_size
    if abs(total - 1.0) > 1e-9:
        raise ValueError(f'As proporções devem somar 1.0, mas somaram {total:.6f}.')


def extract_subject_id(file_path: Path) -> str:
    # Extrai o ID do paciente para impedir vazamento entre conjuntos (OASIS ou ADNI).
    name = file_path.name
    match = OASIS_SUBJECT_PATTERN.search(name)
    if match:
        return match.group(1).upper()
    match = ADNI_SUBJECT_PATTERN.search(name)
    if match:
        return match.group(1)
    raise ValueError(f'Não foi possível extrair subject_id de: {name}')


def collect_dataset_rows(dataset_dir: Path) -> pd.DataFrame:
    # Lê todas as imagens válidas, classe e ID de paciente.
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
                        'subject_id': extract_subject_id(file_path),
                    },
                )

    if not rows:
        raise ValueError('Nenhuma imagem válida encontrada no dataset informado.')
    return pd.DataFrame(rows)


def get_stratify_labels_or_none(labels: pd.Series, stage_name: str) -> pd.Series | None:
    # Usa estratificação só quando há sujeitos suficientes por classe para evitar erro.
    label_counts = labels.value_counts()
    if label_counts.empty:
        return None
    if int(label_counts.min()) < 2:
        print(
            f"Aviso: estratificação desativada em '{stage_name}' "
            f"porque há classe com menos de 2 sujeitos.",
        )
        return None
    return labels


def split_subjects(
    dataset_df: pd.DataFrame,
    train_size: float,
    val_size: float,
    test_size: float,
    seed: int,
) -> pd.DataFrame:
    # Cria tabela de sujeitos únicos + label para estratificar no nível do paciente.
    subject_df = dataset_df[['subject_id', 'label']].drop_duplicates()
    duplicated_subject_labels = subject_df.duplicated(subset=['subject_id'], keep=False)
    if duplicated_subject_labels.any():
        # Evita ambiguidade caso um mesmo sujeito apareça com labels diferentes.
        raise ValueError('Há sujeitos com mais de uma label; revise o dataset.')

    # Primeiro split: treino vs temporário.
    stratify_first = get_stratify_labels_or_none(subject_df['label'], 'train_vs_temp')
    train_subjects, temp_subjects = train_test_split(
        subject_df,
        test_size=(1.0 - train_size),
        random_state=seed,
        stratify=stratify_first,
    )

    # Segundo split: validação vs teste dentro do temporário.
    val_ratio_inside_temp = val_size / (val_size + test_size)
    stratify_second = get_stratify_labels_or_none(temp_subjects['label'], 'val_vs_test')
    val_subjects, test_subjects = train_test_split(
        temp_subjects,
        test_size=(1.0 - val_ratio_inside_temp),
        random_state=seed,
        stratify=stratify_second,
    )

    # Marca cada sujeito com o split correspondente.
    train_subjects = train_subjects.copy()
    val_subjects = val_subjects.copy()
    test_subjects = test_subjects.copy()
    train_subjects['split'] = 'train'
    val_subjects['split'] = 'val'
    test_subjects['split'] = 'test'
    subject_split_df = pd.concat([train_subjects, val_subjects, test_subjects], ignore_index=True)

    # Junta o split por sujeito em todas as imagens.
    split_df = dataset_df.merge(
        subject_split_df[['subject_id', 'split']],
        on='subject_id',
        how='inner',
    )
    return split_df


def print_report(split_df: pd.DataFrame) -> None:
    # Exibe resumo por split/classe e também por quantidade de sujeitos.
    print('\nResumo por split e classe (imagens):\n')
    image_counts = (
        split_df.groupby(['split', 'label'])
        .size()
        .reset_index(name='count')
        .sort_values(by=['split', 'label'])
    )
    print(image_counts.to_string(index=False))

    print('\nResumo por split (sujeitos únicos):\n')
    subject_counts = (
        split_df.groupby('split')['subject_id']
        .nunique()
        .rename('subjects')
        .reset_index()
        .sort_values(by='split')
    )
    print(subject_counts.to_string(index=False))

    print(f'\nTotal de imagens: {len(split_df)}')
    print(f"Total de sujeitos: {split_df['subject_id'].nunique()}")


def verify_no_subject_leakage(split_df: pd.DataFrame) -> None:
    # Confirma que nenhum sujeito aparece em mais de um split.
    per_subject_split_count = split_df.groupby('subject_id')['split'].nunique()
    leakage_count = int((per_subject_split_count > 1).sum())
    if leakage_count > 0:
        raise RuntimeError(
            f'Foram encontrados {leakage_count} sujeitos presentes em mais de um split.',
        )


def main() -> None:
    # Orquestra coleta, split por paciente, validação e gravação do CSV.
    args = parse_args()
    dataset_dir: Path = args.dataset_dir
    output_csv: Path = args.output_csv

    if not dataset_dir.exists():
        raise FileNotFoundError(f'O diretório não existe: {dataset_dir}')
    if not dataset_dir.is_dir():
        raise NotADirectoryError(f'O caminho informado não é uma pasta: {dataset_dir}')

    validate_sizes(args.train_size, args.val_size, args.test_size)
    dataset_df = collect_dataset_rows(dataset_dir)
    split_df = split_subjects(
        dataset_df=dataset_df,
        train_size=args.train_size,
        val_size=args.val_size,
        test_size=args.test_size,
        seed=args.seed,
    )
    verify_no_subject_leakage(split_df)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    split_df.to_csv(output_csv, index=False)

    print_report(split_df)
    print(f'\nCSV gerado em: {output_csv.resolve()}')


if __name__ == '__main__':
    # Mantém execução direta em linha de comando.
    main()
