from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    # Define caminhos dos arquivos de métricas que serão comparados.
    parser = argparse.ArgumentParser(
        description='Compara métricas de diferentes modelos e gera tabela final.',
    )
    parser.add_argument(
        '--baseline-metrics',
        type=Path,
        default=Path('reports_adni/baseline_ml_grouped_binary/metrics.json'),
        help='Arquivo metrics.json do baseline.',
    )
    parser.add_argument(
        '--cnn-metrics',
        type=Path,
        default=Path('reports_adni/cnn_transfer_learning_e5_grouped_binary/metrics.json'),
        help='Arquivo metrics.json da CNN.',
    )
    parser.add_argument(
        '--vit-metrics',
        type=Path,
        default=None,
        help='Arquivo metrics.json do ViT.',
    )
    parser.add_argument(
        '--monai-metrics',
        type=Path,
        default=None,
        help='Arquivo metrics.json do MONAI 2D ResNet18.',
    )
    parser.add_argument(
        '--biomedclip-metrics',
        type=Path,
        default=None,
        help='Arquivo metrics.json do BiomedCLIP embeddings (opcional).',
    )
    parser.add_argument(
        '--output-csv',
        type=Path,
        default=Path('reports_adni/final_comparison/comparison_table.csv'),
        help='Arquivo CSV de saída com comparação.',
    )
    parser.add_argument(
        '--output-md',
        type=Path,
        default=Path('reports_adni/final_comparison/comparison_table.md'),
        help='Arquivo Markdown de saída com comparação.',
    )
    return parser.parse_args()


def load_metrics(metrics_path: Path) -> dict[str, object]:
    # Carrega JSON de métricas com validação de existência.
    if not metrics_path.exists():
        raise FileNotFoundError(f'Arquivo de métricas não encontrado: {metrics_path}')
    with metrics_path.open('r', encoding='utf-8') as metrics_file:
        return json.load(metrics_file)


def safe_float(value: object) -> float | None:
    # Converte valor para float quando possível, para manter consistência na tabela.
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def build_comparison_df(
    baseline_metrics: dict[str, object],
    cnn_metrics: dict[str, object],
    vit_metrics: dict[str, object] | None = None,
    monai_metrics: dict[str, object] | None = None,
    biomedclip_metrics: dict[str, object] | None = None,
) -> pd.DataFrame:
    # Cria DataFrame único para facilitar leitura e export em CSV/Markdown.
    rows = [
        {
            'model': 'Baseline ML',
            'val_accuracy': safe_float(baseline_metrics.get('val_accuracy')),
            'test_accuracy': safe_float(baseline_metrics.get('test_accuracy')),
            'test_loss': safe_float(baseline_metrics.get('test_loss')),
            'notes': 'LogisticRegression com features simples (grayscale + resize)',
        },
        {
            'model': 'CNN Transfer Learning',
            'val_accuracy': safe_float(cnn_metrics.get('best_val_accuracy')),
            'test_accuracy': safe_float(cnn_metrics.get('test_accuracy')),
            'test_loss': safe_float(cnn_metrics.get('test_loss')),
            'notes': 'ResNet18 com fine-tuning em split por paciente',
        },
    ]
    if vit_metrics is not None:
        rows.append(
            {
                'model': 'ViT Transfer Learning',
                'val_accuracy': safe_float(
                    vit_metrics.get('best_val_accuracy', vit_metrics.get('val_accuracy')),
                ),
                'test_accuracy': safe_float(vit_metrics.get('test_accuracy')),
                'test_loss': safe_float(vit_metrics.get('test_loss')),
                'notes': 'ViT-B/16 com transfer learning em split por paciente',
            },
        )
    if monai_metrics is not None:
        rows.append(
            {
                'model': 'MONAI 2D ResNet18',
                'val_accuracy': safe_float(
                    monai_metrics.get('best_val_accuracy', monai_metrics.get('val_accuracy')),
                ),
                'test_accuracy': safe_float(monai_metrics.get('test_accuracy')),
                'test_loss': safe_float(monai_metrics.get('test_loss')),
                'notes': 'MONAI ResNet18 2D sem pretraining em split por paciente',
            },
        )
    if biomedclip_metrics is not None:
        rows.append(
            {
                'model': 'BiomedCLIP Embeddings',
                'val_accuracy': safe_float(
                    biomedclip_metrics.get('val_accuracy', biomedclip_metrics.get('best_val_accuracy')),
                ),
                'test_accuracy': safe_float(biomedclip_metrics.get('test_accuracy')),
                'test_loss': safe_float(biomedclip_metrics.get('test_loss')),
                'notes': 'BiomedCLIP congelado + LogisticRegression balanceada',
            },
        )
    comparison_df = pd.DataFrame(rows)
    return comparison_df


def save_markdown_table(comparison_df: pd.DataFrame, output_md: Path) -> None:
    # Salva a tabela em Markdown para uso direto no relatório final.
    output_md.parent.mkdir(parents=True, exist_ok=True)
    table_df = comparison_df.copy()

    # Formata campos numéricos para leitura mais clara.
    for column in ['val_accuracy', 'test_accuracy', 'test_loss']:
        table_df[column] = table_df[column].apply(
            lambda value: f'{value:.4f}' if pd.notna(value) else '-',
        )

    # Gera Markdown manualmente para evitar dependência opcional do pacote tabulate.
    headers = list(table_df.columns)
    header_line = '| ' + ' | '.join(headers) + ' |'
    separator_line = '| ' + ' | '.join(['---'] * len(headers)) + ' |'
    data_lines = []
    for row_values in table_df.itertuples(index=False, name=None):
        row_line = '| ' + ' | '.join(str(value) for value in row_values) + ' |'
        data_lines.append(row_line)

    markdown_lines = [header_line, separator_line, *data_lines]
    output_md.write_text('\n'.join(markdown_lines) + '\n', encoding='utf-8')


def main() -> None:
    # Orquestra leitura de métricas, criação da comparação e export dos artefatos.
    args = parse_args()

    baseline_metrics = load_metrics(args.baseline_metrics)
    cnn_metrics = load_metrics(args.cnn_metrics)
    vit_metrics = load_metrics(args.vit_metrics) if args.vit_metrics is not None else None
    monai_metrics = load_metrics(args.monai_metrics) if args.monai_metrics is not None else None
    biomedclip_metrics = (
        load_metrics(args.biomedclip_metrics) if args.biomedclip_metrics is not None else None
    )
    comparison_df = build_comparison_df(
        baseline_metrics=baseline_metrics,
        cnn_metrics=cnn_metrics,
        vit_metrics=vit_metrics,
        monai_metrics=monai_metrics,
        biomedclip_metrics=biomedclip_metrics,
    )

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    comparison_df.to_csv(args.output_csv, index=False)
    save_markdown_table(comparison_df=comparison_df, output_md=args.output_md)

    print('\nComparação gerada:')
    print(f'- CSV: {args.output_csv.resolve()}')
    print(f'- MD: {args.output_md.resolve()}')
    print('\nTabela resumo:\n')
    print(comparison_df.to_string(index=False))


if __name__ == '__main__':
    # Mantém execução direta do script em linha de comando.
    main()
