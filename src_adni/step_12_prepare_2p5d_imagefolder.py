from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pydicom
from PIL import Image
from tqdm import tqdm

from step_00_adni_prepare_imagefolder import (
    _pixel_array_to_2d_slice,
    build_dicom_series_index,
    collect_dicom_scan_roots,
    find_dicom_series_dir,
    load_local_adni_config,
    resolve_dicom_root,
    resolve_output_dir,
    sort_dicom_files,
)


IMAGE_ID_PATTERN = re.compile(r'__(?P<image_id>\d+)__visit-')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Gera ImageFolder 2.5D a partir dos DICOM ADNI e do CSV de split existente.',
    )
    parser.add_argument(
        '--split-csv',
        type=Path,
        default=Path('reports_adni/split_assignments_grouped_binary.csv'),
        help='CSV atual com filepath, label, subject_id e split.',
    )
    parser.add_argument(
        '--dicom-root',
        type=Path,
        default=None,
        help='Raiz DICOM principal. Se omitido, usa config_adni.local.json / ADNI_DICOM_ROOT.',
    )
    parser.add_argument(
        '--extra-dicom-root',
        type=Path,
        action='append',
        default=None,
        help='Raiz DICOM extra, repetivel. Util para coorte MRI_1, MRI_2, etc.',
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=None,
        help='Pasta ImageFolder 2.5D. Se omitido, cria irma de processed_imagefolder.',
    )
    parser.add_argument(
        '--output-split-csv',
        type=Path,
        default=Path('reports_adni/split_assignments_grouped_binary_2p5d_o2.csv'),
        help='CSV de split atualizado para apontar aos PNGs 2.5D.',
    )
    parser.add_argument(
        '--manifest-csv',
        type=Path,
        default=Path('reports_adni/prepare_2p5d_full_grouped_binary/manifest.csv'),
        help='Manifesto com DICOM series, indices usados e PNG gerado.',
    )
    parser.add_argument(
        '--summary-json',
        type=Path,
        default=Path('reports_adni/prepare_2p5d_full_grouped_binary/summary.json'),
        help='Resumo da preparacao 2.5D.',
    )
    parser.add_argument(
        '--slice-offset',
        type=int,
        default=2,
        help='Distancia em cortes para canais vizinhos: [mid-offset, mid, mid+offset].',
    )
    parser.add_argument(
        '--skip-missing-dicom',
        action='store_true',
        help='Pula linhas sem pasta DICOM em vez de falhar.',
    )
    parser.add_argument(
        '--max-cases',
        type=int,
        default=0,
        help='Se > 0, processa apenas as primeiras N linhas do split.',
    )
    return parser.parse_args()


def load_split_dataframe(split_csv: Path) -> pd.DataFrame:
    if not split_csv.exists():
        raise FileNotFoundError(f'CSV de split nao encontrado: {split_csv}')

    split_df = pd.read_csv(split_csv)
    required_columns = {'filepath', 'label', 'subject_id', 'split'}
    missing_columns = required_columns.difference(split_df.columns)
    if missing_columns:
        raise ValueError(f'Colunas ausentes no CSV: {sorted(missing_columns)}')
    return split_df.copy()


def infer_image_id(filepath: str) -> str:
    match = IMAGE_ID_PATTERN.search(Path(str(filepath)).name)
    if not match:
        raise ValueError(f'Nao consegui inferir image_id do filepath: {filepath}')
    return match.group('image_id')


def infer_default_output_dir(split_df: pd.DataFrame, slice_offset: int) -> Path:
    config, config_path = load_local_adni_config()
    try:
        base_output = resolve_output_dir(None, config, config_path)
        return base_output.parent / f'{base_output.name}_2p5d_o{slice_offset}'
    except Exception:
        first_path = Path(str(split_df.iloc[0]['filepath']))
        if first_path.parent.name in {'Demented', 'Non Demented'}:
            base_output = first_path.parent.parent
            return base_output.parent / f'{base_output.name}_2p5d_o{slice_offset}'
        return Path(f'processed_imagefolder_2p5d_o{slice_offset}').resolve()


def apply_rescale_if_present(pixel_array: np.ndarray, dataset: Any) -> np.ndarray:
    slope = float(getattr(dataset, 'RescaleSlope', 1.0) or 1.0)
    intercept = float(getattr(dataset, 'RescaleIntercept', 0.0) or 0.0)
    return np.asarray(pixel_array, dtype=np.float32) * slope + intercept


def pixel_array_to_volume(pixel_array: np.ndarray) -> np.ndarray:
    arr = np.squeeze(np.asarray(pixel_array, dtype=np.float32))
    if arr.ndim == 2:
        return arr[np.newaxis, :, :]
    if arr.ndim == 3:
        depth_axis = int(np.argmin(arr.shape))
        volume = np.moveaxis(arr, depth_axis, 0)
        if volume.ndim != 3:
            raise ValueError(f'Volume DICOM invalido apos moveaxis: {arr.shape}')
        return volume
    plane = _pixel_array_to_2d_slice(arr)
    return plane[np.newaxis, :, :]


def read_dicom_series_as_volume(series_dir: Path) -> np.ndarray:
    dicom_files = sorted(p for p in series_dir.glob('*.dcm') if p.is_file())
    if not dicom_files:
        raise FileNotFoundError(f'Nenhum .dcm em {series_dir}')

    if len(dicom_files) == 1:
        dataset = pydicom.dcmread(dicom_files[0], force=True)
        pixels = apply_rescale_if_present(dataset.pixel_array, dataset)
        return pixel_array_to_volume(pixels)

    layers: list[np.ndarray] = []
    expected_shape: tuple[int, int] | None = None
    for dicom_path in sort_dicom_files(series_dir):
        dataset = pydicom.dcmread(dicom_path, force=True)
        plane = _pixel_array_to_2d_slice(dataset.pixel_array)
        plane = apply_rescale_if_present(plane, dataset)
        if expected_shape is None:
            expected_shape = tuple(plane.shape)
        if tuple(plane.shape) != expected_shape:
            raise ValueError(
                f'Serie com shapes inconsistentes em {series_dir}: '
                f'esperado {expected_shape}, recebido {tuple(plane.shape)}',
            )
        layers.append(plane.astype(np.float32))

    if not layers:
        raise ValueError(f'Nenhuma fatia valida em {series_dir}')
    return np.stack(layers, axis=0)


def select_2p5d_indices(depth: int, slice_offset: int) -> list[int]:
    if depth <= 0:
        raise ValueError('Volume sem fatias.')
    mid = depth // 2
    offset = max(0, int(slice_offset))
    return [
        int(np.clip(mid - offset, 0, depth - 1)),
        mid,
        int(np.clip(mid + offset, 0, depth - 1)),
    ]


def volume_to_2p5d_rgb(volume: np.ndarray, slice_offset: int) -> tuple[np.ndarray, list[int]]:
    if volume.ndim != 3:
        raise ValueError(f'Volume esperado em D,H,W; recebido shape={volume.shape}')

    indices = select_2p5d_indices(depth=volume.shape[0], slice_offset=slice_offset)
    selected = volume[indices, :, :].astype(np.float32)

    finite_values = volume[np.isfinite(volume)]
    if finite_values.size == 0:
        raise ValueError('Volume sem valores finitos.')

    p_low, p_high = np.percentile(finite_values, (1.0, 99.0))
    scaled = (np.clip(selected, p_low, p_high) - p_low) / (p_high - p_low + 1e-8)
    rgb = np.moveaxis(np.clip(scaled, 0.0, 1.0), 0, -1)
    return (rgb * 255.0).astype(np.uint8), indices


def build_2p5d_filename(original_filepath: str, slice_offset: int) -> str:
    stem = Path(str(original_filepath)).stem
    new_stem = re.sub(r'__mid$', f'__2p5d_o{slice_offset}', stem)
    if new_stem == stem:
        new_stem = f'{stem}__2p5d_o{slice_offset}'
    return f'{new_stem}.png'


def resolve_dicom_roots(args: argparse.Namespace) -> list[Path]:
    config, config_path = load_local_adni_config()
    dicom_root = resolve_dicom_root(args.dicom_root, config, config_path)
    extra_dicom = list(args.extra_dicom_root) if args.extra_dicom_root else []
    return collect_dicom_scan_roots(
        dicom_root,
        config,
        config_path,
        extra_cli=extra_dicom or None,
    )


def save_summary(summary_path: Path, summary: dict[str, object]) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def main() -> None:
    args = parse_args()
    if args.slice_offset < 0:
        raise ValueError('--slice-offset deve ser >= 0')

    split_df = load_split_dataframe(args.split_csv)
    if args.max_cases > 0:
        split_df = split_df.head(args.max_cases).copy()

    output_dir = args.output_dir or infer_default_output_dir(split_df, args.slice_offset)
    output_dir = output_dir.expanduser().resolve()

    dicom_roots = resolve_dicom_roots(args)
    print('Raizes DICOM indexadas:')
    for root in dicom_roots:
        print(f'- {root}')
    dicom_index = build_dicom_series_index(dicom_roots)
    print(f'Pastas I* encontradas: {len(dicom_index)}')
    print(f'Saida 2.5D: {output_dir}')

    output_rows: list[dict[str, object]] = []
    manifest_rows: list[dict[str, object]] = []
    failures: list[str] = []

    for row in tqdm(split_df.to_dict(orient='records'), desc='DICOM -> PNG 2.5D'):
        image_id = infer_image_id(str(row['filepath']))
        series_dir = find_dicom_series_dir(dicom_roots[0], image_id, dicom_index)
        if series_dir is None:
            message = f'{image_id}: pasta DICOM nao encontrada'
            if args.skip_missing_dicom:
                failures.append(message)
                continue
            raise FileNotFoundError(message)

        label = str(row['label'])
        target_dir = output_dir / label
        target_dir.mkdir(parents=True, exist_ok=True)
        target_path = target_dir / build_2p5d_filename(str(row['filepath']), args.slice_offset)

        try:
            volume = read_dicom_series_as_volume(series_dir)
            rgb_image, indices = volume_to_2p5d_rgb(volume, args.slice_offset)
            Image.fromarray(rgb_image, mode='RGB').save(target_path)
        except Exception as exc:
            message = f'{image_id}: erro ao gerar 2.5D ({exc})'
            if args.skip_missing_dicom:
                failures.append(message)
                continue
            raise RuntimeError(message) from exc

        output_row = dict(row)
        output_row['original_filepath'] = str(row['filepath'])
        output_row['filepath'] = str(target_path)
        output_row['image_id'] = image_id
        output_rows.append(output_row)
        manifest_rows.append(
            {
                'image_id': image_id,
                'subject_id': row.get('subject_id'),
                'label': label,
                'split': row.get('split'),
                'dicom_series_dir': str(series_dir),
                'depth': int(volume.shape[0]),
                'slice_indices': ','.join(str(index) for index in indices),
                'slice_offset': int(args.slice_offset),
                'original_filepath': str(row['filepath']),
                'png_2p5d_path': str(target_path),
            },
        )

    if not output_rows:
        raise RuntimeError('Nenhum PNG 2.5D foi gerado.')

    output_split_df = pd.DataFrame(output_rows)
    args.output_split_csv.parent.mkdir(parents=True, exist_ok=True)
    output_split_df.to_csv(args.output_split_csv, index=False)

    args.manifest_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(manifest_rows).to_csv(args.manifest_csv, index=False)

    summary = {
        'input_split_csv': str(args.split_csv),
        'output_dir': str(output_dir),
        'output_split_csv': str(args.output_split_csv),
        'manifest_csv': str(args.manifest_csv),
        'slice_offset': int(args.slice_offset),
        'rows_input': int(len(split_df)),
        'rows_output': int(len(output_rows)),
        'failures': int(len(failures)),
        'class_counts': output_split_df['label'].value_counts().to_dict(),
        'split_counts': output_split_df['split'].value_counts().to_dict(),
    }
    save_summary(args.summary_json, summary)

    print('\nDataset 2.5D gerado:')
    print(f'- ImageFolder: {output_dir}')
    print(f'- Split CSV: {args.output_split_csv.resolve()}')
    print(f'- Manifesto: {args.manifest_csv.resolve()}')
    print(f'- Resumo: {args.summary_json.resolve()}')
    print(f'- PNGs gerados: {len(output_rows)}')
    if failures:
        print(f'- Falhas ignoradas: {len(failures)}')
        for failure in failures[:10]:
            print(f'  - {failure}')


if __name__ == '__main__':
    main()
