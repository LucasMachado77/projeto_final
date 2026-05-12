from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    # Define caminhos de entrada e saída para conversão de labels em modo binário.
    parser = argparse.ArgumentParser(
        description='Converte labels para binário (Demented vs Non Demented).',
    )
    parser.add_argument(
        '--input-csv',
        type=Path,
        default=Path('reports_adni/split_assignments_grouped.csv'),
        help='CSV de entrada com colunas filepath, label, subject_id e split.',
    )
    parser.add_argument(
        '--output-csv',
        type=Path,
        default=Path('reports_adni/split_assignments_grouped_binary.csv'),
        help='CSV de saída com labels binárias.',
    )
    return parser.parse_args()


def convert_to_binary_label(label: str) -> str:
    # Agrupa classes de demência em um único rótulo (OASIS + pastas já binárias do ADNI).
    normalized = label.strip().lower()
    if normalized in {'demented'}:
        return 'Demented'
    if normalized in {'non demented'}:
        return 'Non Demented'
    if normalized in {'mild dementia', 'moderate dementia'}:
        return 'Demented'
    raise ValueError(f'Label não reconhecida para conversão binária: {label}')


def main() -> None:
    # Executa leitura, conversão de labels e gravação do CSV final.
    args = parse_args()
    input_csv: Path = args.input_csv
    output_csv: Path = args.output_csv

    if not input_csv.exists():
        raise FileNotFoundError(f'Arquivo não encontrado: {input_csv}')

    split_df = pd.read_csv(input_csv)
    required_columns = {'filepath', 'label', 'split'}
    missing_columns = required_columns.difference(split_df.columns)
    if missing_columns:
        raise ValueError(f'Colunas ausentes no CSV: {sorted(missing_columns)}')

    binary_df = split_df.copy()
    binary_df['label_original'] = binary_df['label']
    binary_df['label'] = binary_df['label'].apply(convert_to_binary_label)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    binary_df.to_csv(output_csv, index=False)

    print('\nResumo binário por split e classe:\n')
    summary = (
        binary_df.groupby(['split', 'label'])
        .size()
        .reset_index(name='count')
        .sort_values(by=['split', 'label'])
    )
    print(summary.to_string(index=False))
    print(f'\nTotal de imagens: {len(binary_df)}')
    print(f'CSV binário gerado em: {output_csv.resolve()}')


if __name__ == '__main__':
    # Mantém execução direta do script via linha de comando.
    main()
