from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


MODEL_DIR_DEFAULTS: list[tuple[str, Path]] = [
    ('Baseline ML', Path('reports_adni/baseline_ml_full_grouped_binary')),
    ('CNN Transfer Learning', Path('reports_adni/cnn_transfer_learning_full_reg_grouped_binary')),
    ('ViT Transfer Learning', Path('reports_adni/vit_transfer_learning_full_grouped_binary')),
    ('MONAI 2D ResNet18', Path('reports_adni/monai_2d_resnet18_full_grouped_binary')),
    ('BiomedCLIP Embeddings', Path('reports_adni/biomedclip_embeddings_full_grouped_binary')),
    ('CNN 2.5D ResNet18', Path('reports_adni/cnn_2p5d_resnet18_full_grouped_binary')),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Agrega metricas por classe dos experimentos ADNI.',
    )
    parser.add_argument(
        '--baseline-dir',
        type=Path,
        default=MODEL_DIR_DEFAULTS[0][1],
        help='Pasta com metrics.json, classification_report_test.json e confusion_matrix.csv do baseline.',
    )
    parser.add_argument(
        '--cnn-dir',
        type=Path,
        default=MODEL_DIR_DEFAULTS[1][1],
        help='Pasta de resultados da CNN ResNet18 2D.',
    )
    parser.add_argument(
        '--vit-dir',
        type=Path,
        default=MODEL_DIR_DEFAULTS[2][1],
        help='Pasta de resultados do ViT.',
    )
    parser.add_argument(
        '--monai-dir',
        type=Path,
        default=MODEL_DIR_DEFAULTS[3][1],
        help='Pasta de resultados do MONAI 2D.',
    )
    parser.add_argument(
        '--biomedclip-dir',
        type=Path,
        default=MODEL_DIR_DEFAULTS[4][1],
        help='Pasta de resultados do BiomedCLIP embeddings.',
    )
    parser.add_argument(
        '--cnn-2p5d-dir',
        type=Path,
        default=MODEL_DIR_DEFAULTS[5][1],
        help='Pasta de resultados da CNN 2.5D.',
    )
    parser.add_argument(
        '--output-csv',
        type=Path,
        default=Path('reports_adni/final_comparison_full/class_metrics_table.csv'),
        help='CSV de saida com metricas por classe.',
    )
    parser.add_argument(
        '--output-md',
        type=Path,
        default=Path('reports_adni/final_comparison_full/class_metrics_table.md'),
        help='Markdown de saida com metricas por classe.',
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(f'Arquivo nao encontrado: {path}')
    with path.open('r', encoding='utf-8') as json_file:
        return json.load(json_file)


def safe_float(value: object) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def class_metric(report: dict[str, object], class_name: str, metric_name: str) -> float | None:
    class_block = report.get(class_name)
    if not isinstance(class_block, dict):
        return None
    return safe_float(class_block.get(metric_name))


def aggregate_metric(report: dict[str, object], block_name: str, metric_name: str) -> float | None:
    block = report.get(block_name)
    if not isinstance(block, dict):
        return None
    return safe_float(block.get(metric_name))


def load_confusion_values(model_dir: Path) -> dict[str, int | None]:
    confusion_path = model_dir / 'confusion_matrix.csv'
    if not confusion_path.exists():
        return {
            'tp_demented': None,
            'fp_demented': None,
            'fn_demented': None,
            'tn_non_demented': None,
            'predicted_demented': None,
        }

    confusion_df = pd.read_csv(confusion_path, index_col=0)
    required_labels = {'Demented', 'Non Demented'}
    if not required_labels.issubset(set(confusion_df.index)) or not required_labels.issubset(
        set(confusion_df.columns),
    ):
        return {
            'tp_demented': None,
            'fp_demented': None,
            'fn_demented': None,
            'tn_non_demented': None,
            'predicted_demented': None,
        }

    tp = int(confusion_df.loc['Demented', 'Demented'])
    fn = int(confusion_df.loc['Demented', 'Non Demented'])
    fp = int(confusion_df.loc['Non Demented', 'Demented'])
    tn = int(confusion_df.loc['Non Demented', 'Non Demented'])
    return {
        'tp_demented': tp,
        'fp_demented': fp,
        'fn_demented': fn,
        'tn_non_demented': tn,
        'predicted_demented': tp + fp,
    }


def build_row(model_name: str, model_dir: Path) -> dict[str, object]:
    metrics = load_json(model_dir / 'metrics.json')
    report = load_json(model_dir / 'classification_report_test.json')
    confusion_values = load_confusion_values(model_dir)

    test_accuracy = safe_float(metrics.get('test_accuracy'))
    if test_accuracy is None:
        test_accuracy = safe_float(report.get('accuracy'))

    row: dict[str, object] = {
        'model': model_name,
        'test_acc': test_accuracy,
        'balanced_acc': aggregate_metric(report, 'macro avg', 'recall'),
        'macro_f1': aggregate_metric(report, 'macro avg', 'f1-score'),
        'weighted_f1': aggregate_metric(report, 'weighted avg', 'f1-score'),
        'dem_prec': class_metric(report, 'Demented', 'precision'),
        'dem_recall': class_metric(report, 'Demented', 'recall'),
        'dem_f1': class_metric(report, 'Demented', 'f1-score'),
        'dem_support': class_metric(report, 'Demented', 'support'),
        'non_dem_prec': class_metric(report, 'Non Demented', 'precision'),
        'non_dem_recall': class_metric(report, 'Non Demented', 'recall'),
        'non_dem_f1': class_metric(report, 'Non Demented', 'f1-score'),
        'non_dem_support': class_metric(report, 'Non Demented', 'support'),
    }
    row.update(confusion_values)
    return row


def build_class_metrics_df(args: argparse.Namespace) -> pd.DataFrame:
    model_dirs = [
        ('Baseline ML', args.baseline_dir),
        ('CNN Transfer Learning', args.cnn_dir),
        ('ViT Transfer Learning', args.vit_dir),
        ('MONAI 2D ResNet18', args.monai_dir),
        ('BiomedCLIP Embeddings', args.biomedclip_dir),
        ('CNN 2.5D ResNet18', args.cnn_2p5d_dir),
    ]
    rows = [build_row(model_name, model_dir) for model_name, model_dir in model_dirs]
    return pd.DataFrame(rows)


def format_markdown_value(value: object) -> str:
    if value is None or pd.isna(value):
        return '-'
    if isinstance(value, float):
        if value.is_integer() and abs(value) >= 1:
            return str(int(value))
        return f'{value:.4f}'
    return str(value)


def save_markdown_table(metrics_df: pd.DataFrame, output_md: Path) -> None:
    output_md.parent.mkdir(parents=True, exist_ok=True)
    headers = list(metrics_df.columns)
    header_line = '| ' + ' | '.join(headers) + ' |'
    separator_line = '| ' + ' | '.join(['---'] * len(headers)) + ' |'
    data_lines = []
    for row_values in metrics_df.itertuples(index=False, name=None):
        row_line = '| ' + ' | '.join(format_markdown_value(value) for value in row_values) + ' |'
        data_lines.append(row_line)

    output_md.write_text(
        '\n'.join([header_line, separator_line, *data_lines]) + '\n',
        encoding='utf-8',
    )


def print_key_findings(metrics_df: pd.DataFrame) -> None:
    by_accuracy = metrics_df.sort_values('test_acc', ascending=False).iloc[0]
    by_macro_f1 = metrics_df.sort_values('macro_f1', ascending=False).iloc[0]
    by_demented_recall = metrics_df.sort_values('dem_recall', ascending=False).iloc[0]

    print('\nDestaques:')
    print(f"- Melhor test_acc: {by_accuracy['model']} ({by_accuracy['test_acc']:.4f})")
    print(f"- Melhor macro_f1: {by_macro_f1['model']} ({by_macro_f1['macro_f1']:.4f})")
    print(
        f"- Melhor recall Demented: {by_demented_recall['model']} "
        f"({by_demented_recall['dem_recall']:.4f})",
    )


def main() -> None:
    args = parse_args()
    metrics_df = build_class_metrics_df(args)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(args.output_csv, index=False)
    save_markdown_table(metrics_df, args.output_md)

    print('\nTabela de metricas por classe gerada:')
    print(f'- CSV: {args.output_csv.resolve()}')
    print(f'- MD: {args.output_md.resolve()}')
    print('\nResumo:\n')
    print(metrics_df.to_string(index=False))
    print_key_findings(metrics_df)


if __name__ == '__main__':
    main()
