from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import pandas as pd


VALID_IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Valida leitura/preprocessamento MONAI para imagens ADNI ja preparadas.',
    )
    parser.add_argument(
        '--source',
        choices=('split_csv', 'imagefolder'),
        default='split_csv',
        help='Origem dos dados para o probe: CSV de split ADNI ou pasta ImageFolder.',
    )
    parser.add_argument(
        '--split-csv',
        type=Path,
        default=Path('reports_adni/split_assignments_grouped_binary.csv'),
        help='CSV ADNI com colunas filepath, label e split.',
    )
    parser.add_argument(
        '--dataset-dir',
        type=Path,
        default=None,
        help='Pasta ImageFolder alternativa (classes como subpastas). Usada com --source imagefolder.',
    )
    parser.add_argument(
        '--split',
        default='train',
        help='Split do CSV a inspecionar quando --source split_csv.',
    )
    parser.add_argument(
        '--image-size',
        type=int,
        default=224,
        help='Tamanho final HxW aplicado pelo MONAI Resized.',
    )
    parser.add_argument(
        '--num-samples',
        type=int,
        default=12,
        help='Numero maximo de imagens existentes a carregar no probe.',
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=4,
        help='Batch usado para validar DataLoader MONAI.',
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
    parser.add_argument(
        '--strict-paths',
        action='store_true',
        help='Falha se houver qualquer imagem ausente no split selecionado.',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Semente para amostragem balanceada por classe.',
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('reports_adni/monai_dataset_probe'),
        help='Pasta de saida para resumo JSON/Markdown.',
    )
    return parser.parse_args()


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


def build_class_mapping(labels: list[str]) -> dict[str, int]:
    return {label: index for index, label in enumerate(sorted(labels))}


def select_balanced_rows(split_df: pd.DataFrame, num_samples: int, seed: int) -> pd.DataFrame:
    if num_samples <= 0 or len(split_df) <= num_samples:
        return split_df.copy()

    labels = sorted(split_df['label'].astype(str).unique().tolist())
    per_label = max(1, num_samples // max(len(labels), 1))
    selected_parts: list[pd.DataFrame] = []

    for label in labels:
        label_df = split_df[split_df['label'].astype(str) == label]
        take_n = min(per_label, len(label_df))
        if take_n > 0:
            selected_parts.append(label_df.sample(n=take_n, random_state=seed))

    selected = pd.concat(selected_parts) if selected_parts else split_df.head(0)
    remaining_n = num_samples - len(selected)
    if remaining_n > 0:
        remaining = split_df.drop(index=selected.index, errors='ignore')
        if not remaining.empty:
            selected = pd.concat(
                [
                    selected,
                    remaining.sample(n=min(remaining_n, len(remaining)), random_state=seed),
                ],
            )

    return selected.sort_index().copy()


def load_records_from_split(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not args.split_csv.exists():
        raise FileNotFoundError(f'CSV de split nao encontrado: {args.split_csv}')

    split_df = pd.read_csv(args.split_csv)
    required_columns = {'filepath', 'label', 'split'}
    missing_columns = required_columns.difference(split_df.columns)
    if missing_columns:
        raise ValueError(f'Colunas ausentes no CSV: {sorted(missing_columns)}')

    label_to_id = build_class_mapping(split_df['label'].astype(str).unique().tolist())
    selected_split_df = split_df[split_df['split'].astype(str) == str(args.split)].copy()
    if selected_split_df.empty:
        raise ValueError(f'Nenhuma linha encontrada para split={args.split!r}')

    selected_split_df['resolved_filepath'] = selected_split_df['filepath'].apply(
        lambda value: remap_path(Path(str(value)), args.path_prefix_from, args.path_prefix_to),
    )
    selected_split_df['path_exists'] = selected_split_df['resolved_filepath'].apply(lambda path: path.is_file())

    missing_paths = selected_split_df[~selected_split_df['path_exists']]
    if args.strict_paths and not missing_paths.empty:
        examples = '\n'.join(str(path) for path in missing_paths['resolved_filepath'].head(5))
        raise FileNotFoundError(
            'Ha imagens ausentes no split selecionado. Exemplos:\n'
            f'{examples}\n'
            'Use --path-prefix-from/--path-prefix-to se os dados mudaram de disco, '
            'ou rode sem --strict-paths para testar apenas arquivos existentes.',
        )

    existing_df = selected_split_df[selected_split_df['path_exists']].copy()
    sampled_df = select_balanced_rows(existing_df, args.num_samples, args.seed)
    if sampled_df.empty:
        raise FileNotFoundError(
            'Nenhuma imagem existente foi encontrada para o probe MONAI. '
            'O CSV pode apontar para um disco externo ausente; use --path-prefix-from/--path-prefix-to '
            'ou gere novamente o ImageFolder ADNI com step_00.',
        )

    records = [
        {
            'image': str(row.resolved_filepath),
            'label': int(label_to_id[str(row.label)]),
            'label_text': str(row.label),
            'subject_id': str(getattr(row, 'subject_id', '')),
            'split': str(row.split),
        }
        for row in sampled_df.itertuples(index=False)
    ]

    summary = {
        'source': 'split_csv',
        'split_csv': str(args.split_csv),
        'selected_split': str(args.split),
        'rows_total_csv': int(len(split_df)),
        'rows_selected_split': int(len(selected_split_df)),
        'rows_existing_selected_split': int(len(existing_df)),
        'rows_missing_selected_split': int(len(missing_paths)),
        'sampled_records': int(len(records)),
        'label_to_id': label_to_id,
        'class_counts_selected_split': {
            str(label): int(count)
            for label, count in selected_split_df['label'].astype(str).value_counts().sort_index().items()
        },
        'class_counts_existing_selected_split': {
            str(label): int(count)
            for label, count in existing_df['label'].astype(str).value_counts().sort_index().items()
        },
    }
    return records, summary


def load_records_from_imagefolder(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if args.dataset_dir is None:
        raise ValueError('--dataset-dir e obrigatorio quando --source imagefolder')
    if not args.dataset_dir.is_dir():
        raise NotADirectoryError(f'Dataset dir nao encontrado: {args.dataset_dir}')

    class_dirs = sorted([path for path in args.dataset_dir.iterdir() if path.is_dir()], key=lambda p: p.name.lower())
    if not class_dirs:
        raise ValueError(f'Nenhuma subpasta de classe encontrada em {args.dataset_dir}')

    label_to_id = build_class_mapping([path.name for path in class_dirs])
    rows: list[dict[str, Any]] = []
    class_counts: dict[str, int] = {}
    for class_dir in class_dirs:
        image_paths = sorted(
            path
            for path in class_dir.rglob('*')
            if path.is_file() and path.suffix.lower() in VALID_IMAGE_EXTENSIONS
        )
        class_counts[class_dir.name] = len(image_paths)
        for image_path in image_paths:
            rows.append(
                {
                    'image': str(image_path),
                    'label': int(label_to_id[class_dir.name]),
                    'label_text': class_dir.name,
                    'subject_id': '',
                    'split': 'probe',
                },
            )

    if not rows:
        raise FileNotFoundError(f'Nenhuma imagem valida encontrada em {args.dataset_dir}')

    rows_df = pd.DataFrame(rows)
    sampled_df = select_balanced_rows(rows_df, args.num_samples, args.seed)
    records = sampled_df.to_dict(orient='records')
    summary = {
        'source': 'imagefolder',
        'dataset_dir': str(args.dataset_dir),
        'rows_total_imagefolder': int(len(rows_df)),
        'sampled_records': int(len(records)),
        'label_to_id': label_to_id,
        'class_counts_imagefolder': class_counts,
    }
    return records, summary


def import_monai(output_dir: Path) -> tuple[Any, Any, Any, Any, Any, Any, str, str]:
    # Keep matplotlib cache inside the probe output when MONAI imports plotting helpers.
    mpl_cache = output_dir / '.matplotlib_cache'
    mpl_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault('MPLCONFIGDIR', str(mpl_cache.resolve()))

    try:
        import torch
        import monai
        from monai.data import DataLoader, Dataset
        from monai.transforms import Compose, LoadImaged, Resized, ScaleIntensityd, ToTensord
    except ModuleNotFoundError as exc:
        if exc.name == 'monai':
            raise ModuleNotFoundError(
                'MONAI nao esta instalado. Rode: pip install -r requirements.txt'
            ) from exc
        raise

    return Compose, LoadImaged, Resized, ScaleIntensityd, ToTensord, Dataset, monai.__version__, torch.__version__


def run_monai_probe(
    records: list[dict[str, Any]],
    image_size: int,
    batch_size: int,
    output_dir: Path,
) -> dict[str, Any]:
    (
        Compose,
        LoadImaged,
        Resized,
        ScaleIntensityd,
        ToTensord,
        Dataset,
        monai_version,
        torch_version,
    ) = import_monai(output_dir)
    from monai.data import DataLoader

    transforms = Compose(
        [
            LoadImaged(keys='image', image_only=True, ensure_channel_first=True),
            ScaleIntensityd(keys='image', minv=0.0, maxv=1.0),
            Resized(keys='image', spatial_size=(image_size, image_size), mode='bilinear'),
            ToTensord(keys=('image', 'label')),
        ],
    )
    dataset = Dataset(data=records, transform=transforms)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    batch = next(iter(loader))
    images = batch['image']
    labels = batch['label']

    return {
        'monai_version': monai_version,
        'torch_version': torch_version,
        'batch_size': int(images.shape[0]),
        'image_shape': [int(dim) for dim in images.shape],
        'image_dtype': str(images.dtype),
        'label_shape': [int(dim) for dim in labels.shape],
        'label_dtype': str(labels.dtype),
        'image_min': float(images.min().item()),
        'image_max': float(images.max().item()),
        'image_mean': float(images.mean().item()),
    }


def write_outputs(summary: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / 'probe_summary.json'
    md_path = output_dir / 'probe_summary.md'

    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')

    lines = [
        '# ADNI MONAI dataset probe',
        '',
        f"- source: `{summary.get('source')}`",
        f"- sampled_records: `{summary.get('sampled_records')}`",
        f"- MONAI: `{summary.get('monai', {}).get('monai_version', '-')}`",
        f"- torch: `{summary.get('monai', {}).get('torch_version', '-')}`",
        f"- batch image shape: `{summary.get('monai', {}).get('image_shape', '-')}`",
        f"- intensity range: `{summary.get('monai', {}).get('image_min', '-')}` to "
        f"`{summary.get('monai', {}).get('image_max', '-')}`",
        '',
        '## Class mapping',
        '',
    ]
    for label, label_id in summary.get('label_to_id', {}).items():
        lines.append(f'- `{label}` -> `{label_id}`')

    if summary.get('rows_missing_selected_split') is not None:
        lines.extend(
            [
                '',
                '## Path check',
                '',
                f"- selected split rows: `{summary.get('rows_selected_split')}`",
                f"- existing image paths: `{summary.get('rows_existing_selected_split')}`",
                f"- missing image paths: `{summary.get('rows_missing_selected_split')}`",
            ],
        )

    md_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main() -> None:
    args = parse_args()

    if args.source == 'split_csv':
        records, summary = load_records_from_split(args)
    else:
        records, summary = load_records_from_imagefolder(args)

    summary['image_size'] = int(args.image_size)
    summary['batch_size_requested'] = int(args.batch_size)
    summary['monai'] = run_monai_probe(
        records=records,
        image_size=args.image_size,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
    )
    write_outputs(summary, args.output_dir)

    print('\nProbe MONAI concluido:')
    print(f'- JSON: {(args.output_dir / "probe_summary.json").resolve()}')
    print(f'- MD: {(args.output_dir / "probe_summary.md").resolve()}')
    print(f'- batch image shape: {summary["monai"]["image_shape"]}')
    print(f'- intensidade: {summary["monai"]["image_min"]:.4f} a {summary["monai"]["image_max"]:.4f}')


if __name__ == '__main__':
    main()
