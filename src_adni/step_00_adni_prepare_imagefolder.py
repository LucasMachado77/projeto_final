from __future__ import annotations

# Prepara dataset estilo ImageFolder a partir do ADNI (DICOM + CSVs study_files).
# Harmoniza visitas de imagem vs DXSUM (sc/scmri->bl), fallback por data próxima ala ADNIMERGE
# quando o código de visita não existe no DXSUM (v02, v04, ...).

import argparse
import json
import os
import random
import re
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import pydicom
from PIL import Image
from tqdm import tqdm

# Visitas de imagem LONI que correspondem à baseline clínica "bl" no DXSUM (documentação ADNI comum).
VISIT_ALIASES_TO_DXSUM: dict[str, str] = {
    'sc': 'bl',
    'scmri': 'bl',
}


def _repo_roots_ordered() -> list[Path]:
    # Pastas candidatas à raiz do projeto (ex.: projeto_final ou mei_imagens_medicas_2025).
    src_adni = Path(__file__).resolve().parent
    inner = src_adni.parent
    roots: list[Path] = []
    for candidate in (inner, inner.parent):
        resolved = candidate.resolve()
        if resolved not in [r.resolve() for r in roots]:
            roots.append(candidate)
    return roots


def load_local_adni_config() -> tuple[dict[str, Any], Path | None]:
    # Lê JSON opcional na raiz do projeto (cada máquina o seu; não versionar).
    for root in _repo_roots_ordered():
        for name in ('config_adni.local.json', '.adni_paths.json'):
            path = root / name
            if path.is_file():
                raw = json.loads(path.read_text(encoding='utf-8'))
                if not isinstance(raw, dict):
                    raise ValueError(f'{path} deve ser um objeto JSON (chaves string).')
                return raw, path
    return {}, None


def _cfg_path(
    config: dict[str, Any],
    key: str,
    config_file: Path | None = None,
) -> Path | None:
    # Extrai caminho do config; valores relativos são relativos à pasta do .json.
    value = config.get(key)
    if value is None or value == '':
        return None
    path = Path(str(value)).expanduser()
    if not path.is_absolute() and config_file is not None:
        path = (config_file.parent.resolve() / path).resolve()
    return path


def _external_data_roots(config: dict[str, Any], config_file: Path | None) -> list[Path]:
    # Disco externo ou pasta única de dados (ex.: E:\\data_lc): JSON ou ADNI_DATA_ROOT.
    roots: list[Path] = []
    seen: set[str] = set()

    def add(p: Path) -> None:
        if p.is_dir():
            k = str(p.resolve())
            if k not in seen:
                seen.add(k)
                roots.append(p.resolve())

    env_root = os.environ.get('ADNI_DATA_ROOT', '').strip()
    if env_root:
        add(Path(env_root).expanduser())
    for json_key in ('adni_data_root', 'data_root'):
        p = _cfg_path(config, json_key, config_file)
        if p is not None:
            add(p)
    return roots


def _study_files_dirs(config: dict[str, Any], config_file: Path | None = None) -> list[Path]:
    # Pastas de CSVs da coorte; inclui subpastas sob disco externo (adni_data_root).
    found: list[Path] = []
    extra = _cfg_path(config, 'study_files_dir', config_file)
    if extra is not None and extra.is_dir():
        r = extra.resolve()
        if r not in [p.resolve() for p in found]:
            found.append(r)
    for ext in _external_data_roots(config, config_file):
        for sub in ('study_files', 'ADNI/study_files', ''):
            path = ext / sub if sub else ext
            if path.is_dir() and path.resolve() not in [p.resolve() for p in found]:
                found.append(path.resolve())
    for root in _repo_roots_ordered():
        for sub in ('ADNI/study_files', 'study_files', 'Data/study_files', 'data/study_files'):
            path = root / sub
            if path.is_dir() and path.resolve() not in [p.resolve() for p in found]:
                found.append(path.resolve())
    return found


def _all_dirs_for_shallow_csv_scan(
    config: dict[str, Any],
    config_file: Path | None,
) -> list[Path]:
    # Onde procurar Clinical_*.csv só no nível superior (não rglob no volume inteiro).
    ordered: list[Path] = []
    seen: set[str] = set()

    def add(path: Path) -> None:
        if not path.is_dir():
            return
        k = str(path.resolve())
        if k not in seen:
            seen.add(k)
            ordered.append(path.resolve())

    for d in _study_files_dirs(config, config_file):
        add(d)
    for ext in _external_data_roots(config, config_file):
        add(ext)
        add((ext / 'study_files').resolve())
        add((ext / 'ADNI').resolve())
        add((ext / 'Data').resolve())
    for root in _repo_roots_ordered():
        add(root.resolve())
        add((root / 'ADNI').resolve())
        add((root / 'Data').resolve())
        add((root / 'data').resolve())
    return ordered


def _pick_latest_csv(directory: Path, glob_pattern: str) -> Path | None:
    # Escolhe o ficheiro mais recente por nome (sufixo de data no CSV da coorte).
    matches = sorted(directory.glob(glob_pattern), key=lambda p: p.name)
    return matches[-1] if matches else None


def resolve_key_mri_csv(
    explicit: Path | None,
    config: dict[str, Any],
    config_file: Path | None = None,
) -> Path:
    if explicit is not None:
        path = explicit.expanduser()
        if not path.is_file():
            raise FileNotFoundError(f'CSV Key MRI não encontrado: {path.resolve()}')
        return path.resolve()
    env_v = os.environ.get('ADNI_KEY_MRI_CSV')
    if env_v:
        path = Path(env_v).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f'ADNI_KEY_MRI_CSV não aponta para ficheiro válido: {path}')
        return path.resolve()
    cfg_p = _cfg_path(config, 'key_mri_csv', config_file)
    if cfg_p is not None:
        if not cfg_p.is_file():
            raise FileNotFoundError(f'key_mri_csv no config não existe: {cfg_p.resolve()}')
        return cfg_p.resolve()
    search_dirs = _all_dirs_for_shallow_csv_scan(config, config_file)
    for study_dir in search_dirs:
        hit = _pick_latest_csv(study_dir, 'Clinical_T1w_Imaging_Cohort_Key_MRI_*.csv')
        if hit is not None:
            return hit.resolve()
    tried = '\n  '.join(str(p) for p in search_dirs) if search_dirs else '(nenhuma)'
    msg = (
        'CSV Key MRI não encontrado. Use --key-mri-csv, ADNI_KEY_MRI_CSV, key_mri_csv ou '
        'adni_data_root / ADNI_DATA_ROOT (ex.: E:/data_lc com CSV na raiz ou em study_files).\n'
        f'Pastas verificadas (ficheiros .csv só na raiz de cada uma):\n  {tried}'
    )
    raise FileNotFoundError(msg)


def resolve_dxsum_csv(
    explicit: Path | None,
    config: dict[str, Any],
    config_file: Path | None = None,
) -> Path:
    if explicit is not None:
        path = explicit.expanduser()
        if not path.is_file():
            raise FileNotFoundError(f'CSV DXSUM não encontrado: {path.resolve()}')
        return path.resolve()
    env_v = os.environ.get('ADNI_DXSUM_CSV')
    if env_v:
        path = Path(env_v).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f'ADNI_DXSUM_CSV não aponta para ficheiro válido: {path}')
        return path.resolve()
    cfg_p = _cfg_path(config, 'dxsum_csv', config_file)
    if cfg_p is not None:
        if not cfg_p.is_file():
            raise FileNotFoundError(f'dxsum_csv no config não existe: {cfg_p.resolve()}')
        return cfg_p.resolve()
    search_dirs = _all_dirs_for_shallow_csv_scan(config, config_file)
    for study_dir in search_dirs:
        hit = _pick_latest_csv(study_dir, 'Clinical_T1w_Imaging_Cohort_DXSUM_*.csv')
        if hit is not None:
            return hit.resolve()
    tried = '\n  '.join(str(p) for p in search_dirs)
    raise FileNotFoundError(
        'CSV DXSUM não encontrado. Defina --dxsum-csv, ADNI_DXSUM_CSV, dxsum_csv ou '
        f'garanta os CSV sob adni_data_root / ADNI_DATA_ROOT.\nPastas:\n  {tried}',
    )


def _first_existing_dir(candidates: list[Path]) -> Path | None:
    for candidate in candidates:
        if candidate.is_dir():
            return candidate.resolve()
    return None


# Pastas LONI ao descompactar a coorte em vários ZIPs (MRI_1, MRI_2, ...).
COHORT_MRI_DIR_GLOB = 'Clinical_T1w_Imaging_Cohort_MRI_*'


def _discover_cohort_mri_dirs(base: Path) -> list[Path]:
    # Lista subpastas tipo Clinical_T1w_Imaging_Cohort_MRI_1 / _2 na raiz `base`.
    if not base.is_dir():
        return []
    return sorted(p.resolve() for p in base.glob(COHORT_MRI_DIR_GLOB) if p.is_dir())


def _resolve_config_path_str(value: str, config_file: Path | None) -> Path:
    # Caminho a partir de string no JSON (relativo = pasta do ficheiro de config).
    path = Path(str(value).strip()).expanduser()
    if not path.is_absolute() and config_file is not None:
        path = (config_file.parent.resolve() / path).resolve()
    return path.resolve()


def _prune_scan_roots(roots: list[Path]) -> list[Path]:
    # Mantém só raízes “externas”: remove pastas que já estão sob outra raiz (evita rglob duplicado).
    resolved = [r for r in (p.resolve() for p in roots) if r.is_dir()]
    resolved.sort(key=lambda p: (len(p.parts), str(p)))
    kept: list[Path] = []
    for p in resolved:
        if any(p == q or _is_path_under(p, q) for q in kept):
            continue
        kept = [q for q in kept if not (_is_path_under(q, p))]
        kept.append(p)
    return sorted(kept, key=lambda x: str(x))


def _is_path_under(child: Path, ancestor: Path) -> bool:
    # True se `child` é subcaminho de `ancestor` (compatível sem is_relative_to em versões antigas).
    try:
        child.relative_to(ancestor.resolve())
        return True
    except ValueError:
        return False


def collect_dicom_scan_roots(
    primary: Path,
    config: dict[str, Any],
    config_file: Path | None,
    extra_cli: list[Path] | None = None,
) -> list[Path]:
    # Junta raiz principal, extras do CLI/config e pastas MRI_1/MRI_2... (coorte em vários ZIPs LONI).
    candidates: list[Path] = [primary.resolve()]

    if extra_cli:
        for p in extra_cli:
            candidates.append(p.expanduser().resolve())

    raw_list = config.get('dicom_roots')
    if isinstance(raw_list, list):
        for item in raw_list:
            if isinstance(item, str) and item.strip():
                cp = _resolve_config_path_str(item, config_file)
                candidates.append(cp)

    for base in (
        primary.resolve(),
        primary.resolve().parent,
        *_external_data_roots(config, config_file),
    ):
        candidates.extend(_discover_cohort_mri_dirs(base))

    return _prune_scan_roots(candidates)


def resolve_dicom_root(
    explicit: Path | None,
    config: dict[str, Any],
    config_file: Path | None = None,
) -> Path:
    if explicit is not None:
        path = explicit.expanduser().resolve()
        if not path.is_dir():
            raise NotADirectoryError(f'DICOM root inexistente ou não é pasta: {path}')
        return path
    env_v = os.environ.get('ADNI_DICOM_ROOT', '').strip()
    if env_v:
        path = Path(env_v).expanduser().resolve()
        if not path.is_dir():
            raise NotADirectoryError(f'ADNI_DICOM_ROOT não é pasta válida: {path}')
        return path
    cfg_p = _cfg_path(config, 'dicom_root', config_file)
    if cfg_p is not None:
        path = cfg_p.resolve()
        if not path.is_dir():
            raise NotADirectoryError(f'dicom_root no config não é pasta: {path}')
        return path
    for ext in _external_data_roots(config, config_file):
        hit = _first_existing_dir(
            [
                ext,
                ext / 'Data',
                ext / 'ADNI' / 'Data',
            ],
        )
        if hit is not None:
            return hit
    for root in _repo_roots_ordered():
        for sub in ('ADNI/Data', 'Data'):
            candidate = (root / sub).resolve()
            if candidate.is_dir():
                return candidate
    raise FileNotFoundError(
        'Pasta DICOM não encontrada. No disco externo use dicom_root ou adni_data_root '
        '(ou ADNI_DATA_ROOT / ADNI_DICOM_ROOT) em config_adni.local.json.',
    )


def resolve_output_dir(
    explicit: Path | None,
    config: dict[str, Any],
    config_file: Path | None = None,
) -> Path:
    if explicit is not None:
        return explicit.expanduser().resolve()
    env_v = os.environ.get('ADNI_OUTPUT_DIR', '').strip()
    if env_v:
        return Path(env_v).expanduser().resolve()
    cfg_p = _cfg_path(config, 'output_dir', config_file)
    if cfg_p is not None:
        return cfg_p.resolve()
    for ext in _external_data_roots(config, config_file):
        out = ext / 'processed_imagefolder'
        if ext.is_dir():
            return out.resolve()
    root = _repo_roots_ordered()[0]
    if (root / 'ADNI').is_dir():
        return (root / 'ADNI' / 'processed_imagefolder').resolve()
    return (root / 'processed_imagefolder').resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='ADNI: DICOM + CSVs -> pastas Non Demented / Demented com PNGs para o pipeline.',
        epilog=(
            'Precedência: argumentos > env (ADNI_DATA_ROOT, ADNI_DICOM_ROOT, ADNI_OUTPUT_DIR, '
            'ADNI_KEY_MRI_CSV, ADNI_DXSUM_CSV) > config_adni.local.json (adni_data_root, etc.).'
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        '--key-mri-csv',
        type=Path,
        default=None,
        help=(
            'CSV Key MRI. Se omitido, procura Clinical_T1w_Imaging_Cohort_Key_MRI_*.csv '
            'em ADNI/study_files ou study_files relativamente ao projeto.'
        ),
    )
    parser.add_argument(
        '--dxsum-csv',
        type=Path,
        default=None,
        help=(
            'CSV DXSUM. Se omitido, procura Clinical_T1w_Imaging_Cohort_DXSUM_*.csv '
            'nas mesmas pastas que Key MRI.'
        ),
    )
    parser.add_argument(
        '--dicom-root',
        type=Path,
        default=None,
        help=(
            'Pasta com DICOM descompactados. Se omitido, usa ADNI/Data ou Data na raiz do projeto.'
        ),
    )
    parser.add_argument(
        '--extra-dicom-root',
        type=Path,
        action='append',
        default=None,
        help=(
            'Raiz DICOM extra (repita a flag). Útil para Clinical_T1w_Imaging_Cohort_MRI_2 '
            'se só MRI_1 estiver em --dicom-root.'
        ),
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=None,
        help=(
            'Saída ImageFolder. Se omitido, ADNI/processed_imagefolder ou processed_imagefolder.'
        ),
    )
    parser.add_argument(
        '--scheme',
        choices=('cn_mci_vs_ad', 'cn_vs_ad'),
        default='cn_mci_vs_ad',
        help=(
            'cn_mci_vs_ad: CN(1)+MCI(2) -> Non Demented, AD(3) -> Demented. '
            'cn_vs_ad: só CN vs AD (MCI excluído).'
        ),
    )
    parser.add_argument(
        '--max-cases',
        type=int,
        default=0,
        help='Se > 0, processa no máximo N linhas (útil para teste rápido).',
    )
    parser.add_argument(
        '--manifest-csv',
        type=Path,
        default=Path('reports_adni/adni_prepare_manifest.csv'),
        help='CSV com image_id, subject_id, dx, png_path, método de cruzamento clínico.',
    )
    parser.add_argument(
        '--merge-stats-txt',
        type=Path,
        default=Path('reports_adni/adni_merge_stats.txt'),
        help='Resumo textual do cruzamento Key MRI x DXSUM.',
    )
    parser.add_argument(
        '--nearest-days',
        type=int,
        default=120,
        help=(
            'Máximo de dias entre image_date (Key) e EXAMDATE (DXSUM) para atribuir '
            'diagnóstico por vizinho temporal (visitas só de imagem como v02/v04).'
        ),
    )
    parser.add_argument(
        '--no-nearest-date',
        action='store_true',
        help='Desliga o fallback por data (só usa codigo de visita harmonizado).',
    )
    parser.add_argument(
        '--keep-repeat-series',
        action='store_true',
        help='Por defeito remove linhas cuja series_description contém REPEAT; use isto para manter.',
    )
    parser.add_argument(
        '--skip-missing-dicom',
        action='store_true',
        help=(
            'Antes do loop PNG, remove do DataFrame linhas cujo image_id não tem pasta I* no disco. '
            'Recomendado quando o CSV é coorte completa mas só parte dos volumes foi baixada (LONI).'
        ),
    )
    parser.add_argument(
        '--balance-output',
        choices=('none', 'downsample'),
        default='none',
        help=(
            'Após --skip-missing-dicom (recomendado): iguala contagens por pasta de classe à menor, '
            'antes de gerar PNG (ImageFolder ~50/50 em nº de imagens; descarta excesso na maioria).'
        ),
    )
    parser.add_argument(
        '--max-rows-per-subject',
        type=int,
        default=0,
        help=(
            'Se > 0, antes de --balance-output limita quantas séries por subject_id entram (amostra aleatória). '
            'Reduz domínio de poucos doentes com muitas imagens na classe majoritária.'
        ),
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Semente para amostragem (--balance-output, --max-rows-per-subject).',
    )
    parser.add_argument(
        '--clean-output-classes',
        action='store_true',
        help=(
            'Antes de gerar PNG, apaga *.png em output_dir/Non Demented e Demented (refazer dataset '
            'sem misturar com uma corrida anterior).'
        ),
    )
    return parser.parse_args()


def diagnosis_to_label(diagnosis: float | int, scheme: str) -> str | None:
    # Converte código DIAGNOSIS do DXSUM (1=CN, 2=MCI, 3=demência) em pasta do pipeline.
    if pd.isna(diagnosis):
        return None
    code = int(diagnosis)
    if scheme == 'cn_mci_vs_ad':
        if code == 3:
            return 'Demented'
        if code in (1, 2):
            return 'Non Demented'
        return None
    if scheme == 'cn_vs_ad':
        if code == 3:
            return 'Demented'
        if code == 1:
            return 'Non Demented'
        return None
    return None


# Pastas de série DICOM no padrão LONI: nome exatamente I seguido só de dígitos (ex.: I28561).
I_SERIES_DIR_PATTERN = re.compile(r'^I(\d+)$')


def build_dicom_series_index(dicom_roots: Path | Sequence[Path]) -> dict[str, Path]:
    # Uma passagem por cada raiz — índice global image_id -> pasta I* (prefere caminho mais curto).
    roots = [dicom_roots] if isinstance(dicom_roots, Path) else list(dicom_roots)
    index: dict[str, Path] = {}
    for dicom_root in roots:
        if not dicom_root.is_dir():
            continue
        for path in dicom_root.rglob('I*'):
            if not path.is_dir():
                continue
            match = I_SERIES_DIR_PATTERN.match(path.name)
            if not match:
                continue
            image_id = match.group(1)
            path_len = len(str(path.resolve()))
            if image_id not in index or path_len < len(str(index[image_id].resolve())):
                index[image_id] = path
    return index


def find_dicom_series_dir(
    dicom_root: Path,
    image_id: str,
    index: dict[str, Path] | None = None,
) -> Path | None:
    # Localiza pasta I{image_id}; use sempre `index` preenchido (build_dicom_series_index).
    key = str(image_id).strip()
    if index is not None:
        return index.get(key)
    folder_name = f'I{key}'
    matches = [p for p in dicom_root.rglob(folder_name) if p.is_dir()]
    if not matches:
        return None
    matches.sort(key=lambda p: len(str(p)))
    return matches[0]


def sort_dicom_files(series_dir: Path) -> list[Path]:
    # Ordena ficheiros DICOM por InstanceNumber para montar volume coerente.
    dcms = [p for p in series_dir.iterdir() if p.is_file() and p.suffix.lower() == '.dcm']
    scored: list[tuple[int, Path]] = []
    for path in dcms:
        try:
            ds = pydicom.dcmread(path, stop_before_pixels=True, force=True)
            inst = int(getattr(ds, 'InstanceNumber', 0) or 0)
        except Exception:
            inst = 0
        scored.append((inst, path))
    scored.sort(key=lambda t: t[0])
    return [p for _, p in scored]


def _pixel_array_to_2d_slice(pixels: np.ndarray) -> np.ndarray:
    # Uniformiza pixel_array do pydicom para matriz 2D (PIL / percentis esperam HxW).
    # Trata multi-frame, singletons (ex. 1x1x256) e volumes 3D (fatia central num eixo).
    arr = np.asarray(pixels, dtype=np.float32)
    a = np.squeeze(arr)
    if a.ndim == 0:
        raise ValueError('pixel_array degenerado (escalar).')
    if a.ndim == 2:
        return a
    if a.ndim == 1:
        # Ex.: série 1×1×N (strip 1D) — uma linha deixamos como imagem 1×N
        return np.expand_dims(a, axis=0)
    if a.ndim == 3:
        # Heurística: fatia central ao longo do eixo mais curto (típico: nº de cortes vs H×W).
        depth_axis = int(np.argmin(a.shape))
        mid = a.shape[depth_axis] // 2
        sl = [slice(None), slice(None), slice(None)]
        sl[depth_axis] = mid
        out = np.asarray(a[tuple(sl)], dtype=np.float32)
        out = np.squeeze(out)
        if out.ndim != 2:
            # Fallback: colapsar mais um singleton
            out = np.squeeze(out)
        if out.ndim == 1:
            return np.expand_dims(out, axis=0)
        if out.ndim == 2:
            return out
        raise ValueError(f'Não foi possível reduzir pixel_array 3D a 2D: shape original {arr.shape}.')
    # 4D+ (ex.: multiframe com canais): remove eixos unitários até 3D
    b = np.asarray(arr, dtype=np.float32)
    while b.ndim > 3:
        b = np.squeeze(b)
        if b.ndim < 3:
            break
    if b.ndim == 3:
        return _pixel_array_to_2d_slice(b)
    raise ValueError(f'Forma de pixel_array não suportada: {pixels.shape}')


def volume_middle_slice_2d(series_dir: Path) -> np.ndarray:
    # Lê a série, empilha e devolve fatia 2D do meio (uint8 para PNG).
    files = list(series_dir.glob('*.dcm'))
    if not files:
        raise FileNotFoundError(f'Nenhum .dcm em {series_dir}')

    if len(files) == 1:
        ds = pydicom.dcmread(files[0], force=True)
        arr = _pixel_array_to_2d_slice(ds.pixel_array)
    else:
        ordered = sort_dicom_files(series_dir)
        if not ordered:
            raise FileNotFoundError(f'Lista DICOM vazia após ordenação em {series_dir}')
        layers: list[np.ndarray] = []
        for fp in ordered:
            ds = pydicom.dcmread(fp, force=True)
            plane = _pixel_array_to_2d_slice(ds.pixel_array)
            if hasattr(ds, 'RescaleSlope') or hasattr(ds, 'RescaleIntercept'):
                slope = float(getattr(ds, 'RescaleSlope', 1.0) or 1.0)
                intercept = float(getattr(ds, 'RescaleIntercept', 0.0) or 0.0)
                plane = plane * slope + intercept
            layers.append(plane)
        vol = np.stack(layers, axis=0)
        mid = vol.shape[0] // 2
        arr = vol[mid]

    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        raise ValueError('Volume sem valores finitos.')
    p_low, p_high = np.percentile(finite, (1.0, 99.0))
    clipped = np.clip(arr, p_low, p_high)
    scaled = (clipped - p_low) / (p_high - p_low + 1e-8)
    scaled = np.clip(scaled, 0.0, 1.0)
    return (scaled * 255.0).astype(np.uint8)


def _fill_nearest_clinical_diagnosis(
    merged: pd.DataFrame,
    dx_clin: pd.DataFrame,
    nearest_days: int,
) -> None:
    # Para linhas sem DIAGNOSIS após merge por visita, usa visita DXSUM mais próxima no tempo.
    still = merged['DIAGNOSIS_NUM'].isna()
    for idx in merged.index[still]:
        subject_id = merged.at[idx, 'subject_id']
        img_dt = merged.at[idx, 'image_date_dt']
        if pd.isna(img_dt):
            merged.at[idx, 'match_method'] = 'sem_data_imagem'
            continue
        cand = dx_clin[
            (dx_clin['PTID'] == subject_id)
            & dx_clin['EXAMDATE_DT'].notna()
            & dx_clin['DIAGNOSIS_NUM'].notna()
        ]
        if cand.empty:
            merged.at[idx, 'match_method'] = 'sem_linha_dxsum'
            continue
        delta_days = (cand['EXAMDATE_DT'] - img_dt).abs().dt.days.astype(float)
        j = int(delta_days.idxmin())
        best_days = float(delta_days.loc[j])
        if best_days <= float(nearest_days):
            merged.at[idx, 'DIAGNOSIS_NUM'] = cand.loc[j, 'DIAGNOSIS_NUM']
            merged.at[idx, 'dx_matched_viscode'] = cand.loc[j, 'VISCODE']
            merged.at[idx, 'match_method'] = f'proxima_data_{int(best_days)}d'
        else:
            merged.at[idx, 'match_method'] = f'sem_match_{int(best_days)}d_longe'


def build_merged_table(
    key_csv: Path,
    dxsum_csv: Path,
    scheme: str,
    *,
    nearest_days: int,
    use_nearest_date: bool,
    exclude_repeat_series: bool,
) -> pd.DataFrame:
    # Cruza Key MRI com DXSUM: alias de visita (imagem vs clínica) + fallback temporal.
    key_df = pd.read_csv(key_csv, dtype=str)
    dx_df = pd.read_csv(dxsum_csv, dtype=str)

    key_df.columns = key_df.columns.str.strip().str.strip('"')
    dx_df.columns = dx_df.columns.str.strip().str.strip('"')

    n_key_in = len(key_df)

    if exclude_repeat_series and 'series_description' in key_df.columns:
        desc_upper = key_df['series_description'].fillna('').str.upper()
        key_df = key_df[~desc_upper.str.contains('REPEAT', na=False)].copy()

    dx_clin = dx_df[['PTID', 'VISCODE', 'EXAMDATE', 'DIAGNOSIS']].copy()
    dx_clin['DIAGNOSIS_NUM'] = pd.to_numeric(dx_clin['DIAGNOSIS'], errors='coerce')
    dx_clin['EXAMDATE_DT'] = pd.to_datetime(dx_clin['EXAMDATE'], errors='coerce')

    key_df['image_date_dt'] = pd.to_datetime(key_df['image_date'], errors='coerce')
    key_df['visit_for_dx'] = key_df['image_visit'].replace(VISIT_ALIASES_TO_DXSUM)

    merged = key_df.merge(
        dx_clin,
        left_on=['subject_id', 'visit_for_dx'],
        right_on=['PTID', 'VISCODE'],
        how='left',
    )

    merged['match_method'] = pd.Series(pd.NA, index=merged.index, dtype='string')
    merged.loc[merged['DIAGNOSIS_NUM'].notna(), 'match_method'] = 'visita_codigo_dxsum'

    merged['dx_matched_viscode'] = pd.Series(pd.NA, index=merged.index, dtype='string')
    vis_ok = merged['DIAGNOSIS_NUM'].notna()
    merged.loc[vis_ok, 'dx_matched_viscode'] = merged.loc[vis_ok, 'VISCODE'].astype('string')

    if use_nearest_date:
        _fill_nearest_clinical_diagnosis(merged, dx_clin, nearest_days)

    merged['match_method'] = merged['match_method'].fillna('sem_diagnostico')
    merged['folder_label'] = merged['DIAGNOSIS_NUM'].apply(
        lambda x: diagnosis_to_label(x, scheme),
    )
    match_breakdown = merged['match_method'].value_counts()

    out = merged[merged['folder_label'].notna()].drop_duplicates(subset=['image_id']).copy()
    out.attrs['n_key_rows_input'] = n_key_in
    out.attrs['n_key_after_repeat_filter'] = len(key_df)
    out.attrs['match_method_counts'] = match_breakdown.to_dict()
    out.attrs['match_method_table'] = match_breakdown.to_string()
    return out


def _write_merge_stats(
    merged: pd.DataFrame,
    stats_path: Path,
    *,
    scheme: str,
    nearest_days: int,
    use_nearest_date: bool,
    exclude_repeat_series: bool,
) -> None:
    # Grava resumo reprodutível do cruzamento clínico (para o relatório / metodologia).
    n_in = merged.attrs.get('n_key_rows_input', '?')
    n_rep = merged.attrs.get('n_key_after_repeat_filter', '?')
    match_tbl = merged.attrs.get('match_method_table', '(sem tabela)')
    lines: list[str] = [
        'ADNI - cruzamento Clinical Key MRI x DXSUM',
        f'Esquema de classes: {scheme}',
        f'Alias de visita (imagem->DXSUM): {VISIT_ALIASES_TO_DXSUM}',
        f'Fallback por data: {"ligado" if use_nearest_date else "desligado"} '
        f'(janela ±{nearest_days} dias)',
        f'Exclui séries REPEAT no nome: {exclude_repeat_series}',
        f'Linhas na Key (entrada): {n_in}',
        f'Linhas na Key após filtro REPEAT: {n_rep}',
        f'Linhas finais com label usável (após scheme + dedup image_id): {len(merged)}',
        '',
        'Contagem por match_method (todas as linhas com DIAGNOSIS mapeável, antes do filtro scheme):',
        match_tbl,
    ]
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def _append_step00_prepare_notes(stats_path: Path, notes: list[str]) -> None:
    # Regista no ficheiro de estatísticas os filtros aplicados após o merge (reprodutível).
    if not notes:
        return
    block = '\n---\nFiltros pós-merge nesta execução do step_00:\n' + '\n'.join(notes) + '\n'
    if stats_path.is_file():
        prev = stats_path.read_text(encoding='utf-8')
        stats_path.write_text(prev + block, encoding='utf-8')
    else:
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        stats_path.write_text(block.lstrip('\n'), encoding='utf-8')


def _cap_rows_per_subject_merged(merged: pd.DataFrame, max_per_subject: int, seed: int) -> pd.DataFrame:
    # Limita quantas linhas (image_id) por paciente entram na preparação dos PNG.
    rng = random.Random(seed)
    chunks: list[pd.DataFrame] = []
    for _, group in merged.groupby('subject_id', sort=False):
        if len(group) <= max_per_subject:
            chunks.append(group)
            continue
        positions = rng.sample(range(len(group)), k=max_per_subject)
        chunks.append(group.iloc[sorted(positions)])
    return pd.concat(chunks, ignore_index=True)


def _balance_folder_labels_downsample(merged: pd.DataFrame, seed: int) -> pd.DataFrame:
    # Iguala nº de linhas entre folder_label (Demented / Non Demented) à classe menor.
    col = 'folder_label'
    counts = merged[col].value_counts()
    if len(counts) < 2:
        return merged
    target_n = int(counts.min())
    parts: list[pd.DataFrame] = []
    for label in counts.index:
        g = merged[merged[col] == label]
        if len(g) <= target_n:
            parts.append(g)
        else:
            parts.append(g.sample(n=target_n, random_state=seed, replace=False))
    return pd.concat(parts, ignore_index=True)


def main() -> None:
    args = parse_args()
    # Precedência: argumentos CLI > variáveis de ambiente > config_adni.local.json > deteção automática.
    adni_config, adni_config_path = load_local_adni_config()
    if adni_config_path is not None:
        print(f'Config local: {adni_config_path.resolve()}')

    args.key_mri_csv = resolve_key_mri_csv(args.key_mri_csv, adni_config, adni_config_path)
    args.dxsum_csv = resolve_dxsum_csv(args.dxsum_csv, adni_config, adni_config_path)
    args.dicom_root = resolve_dicom_root(args.dicom_root, adni_config, adni_config_path)
    args.output_dir = resolve_output_dir(args.output_dir, adni_config, adni_config_path)
    extra_dicom = list(args.extra_dicom_root) if args.extra_dicom_root else []
    dicom_scan_roots = collect_dicom_scan_roots(
        args.dicom_root,
        adni_config,
        adni_config_path,
        extra_cli=extra_dicom or None,
    )
    print('Caminhos resolvidos:')
    print(f'  key_mri:   {args.key_mri_csv}')
    print(f'  dxsum:     {args.dxsum_csv}')
    print(f'  dicom:     {args.dicom_root}  (principal; ver raízes de varredura abaixo)')
    print(f'  output:    {args.output_dir}')
    print('Raízes DICOM indexadas (MRI_1 + MRI_2 + ... / extras):')
    for r in dicom_scan_roots:
        print(f'    - {r}')

    exclude_repeat = not args.keep_repeat_series
    merged = build_merged_table(
        args.key_mri_csv,
        args.dxsum_csv,
        args.scheme,
        nearest_days=args.nearest_days,
        use_nearest_date=not args.no_nearest_date,
        exclude_repeat_series=exclude_repeat,
    )

    _write_merge_stats(
        merged,
        args.merge_stats_txt,
        scheme=args.scheme,
        nearest_days=args.nearest_days,
        use_nearest_date=not args.no_nearest_date,
        exclude_repeat_series=exclude_repeat,
    )

    if args.max_cases > 0:
        merged = merged.head(args.max_cases).copy()

    print(f'A construir índice DICOM em {len(dicom_scan_roots)} raiz(es) (uma passagem cada)...')
    dicom_index = build_dicom_series_index(dicom_scan_roots)
    print(f'Índice DICOM: {len(dicom_index)} pastas I* encontradas.')

    if args.skip_missing_dicom:
        before = len(merged)
        id_str = merged['image_id'].astype(str).str.strip()
        merged = merged[id_str.isin(dicom_index.keys())].copy()
        print(f'--skip-missing-dicom: {before} -> {len(merged)} linhas (só com volume no disco).')

    run_notes: list[str] = []
    if args.max_rows_per_subject > 0:
        before = len(merged)
        merged = _cap_rows_per_subject_merged(merged, args.max_rows_per_subject, args.seed)
        line = f'max-rows-per-subject={args.max_rows_per_subject}: {before} -> {len(merged)} linhas.'
        print(line)
        run_notes.append(line)
    if args.balance_output == 'downsample':
        before_ct = merged['folder_label'].value_counts().to_dict()
        merged = _balance_folder_labels_downsample(merged, args.seed)
        after_ct = merged['folder_label'].value_counts().to_dict()
        line = f'balance-output=downsample: {before_ct} -> {after_ct} | {len(merged)} linhas no loop PNG.'
        print(line)
        run_notes.append(line)
    if run_notes:
        _append_step00_prepare_notes(args.merge_stats_txt, run_notes)

    out_non = args.output_dir / 'Non Demented'
    out_dem = args.output_dir / 'Demented'
    if args.clean_output_classes:
        # Evita misturar PNGs de uma execução anterior com contagens ou filtros diferentes.
        for sub in (out_non, out_dem):
            if sub.is_dir():
                png_list = [p for p in sub.glob('*.png') if p.is_file()]
                for p in png_list:
                    p.unlink()
                print(f'--clean-output-classes: {len(png_list)} .png removidos em {sub}.')
    out_non.mkdir(parents=True, exist_ok=True)
    out_dem.mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict[str, Any]] = []
    ok = 0
    skipped: list[str] = []

    for _, row in tqdm(merged.iterrows(), total=len(merged), desc='ADNI -> PNG'):
        image_id = str(row['image_id']).strip()
        subject_id = str(row['subject_id']).strip()
        label = str(row['folder_label'])
        dx = row['DIAGNOSIS_NUM']

        series_dir = find_dicom_series_dir(args.dicom_root, image_id, dicom_index)
        if series_dir is None:
            roots_hint = ', '.join(str(r) for r in dicom_scan_roots[:3])
            skipped.append(f'{image_id} sem pasta I* nas raízes indexadas ({roots_hint}...)')
            continue

        dest_root = out_dem if label == 'Demented' else out_non
        safe_visit = re.sub(r'[^\w\-]+', '_', str(row.get('image_visit', 'na')))
        png_name = f'{subject_id}__{image_id}__visit-{safe_visit}__mid.png'
        out_path = dest_root / png_name

        try:
            slice_u8 = volume_middle_slice_2d(series_dir)
            Image.fromarray(slice_u8).save(out_path)
        except Exception as exc:
            skipped.append(f'{image_id} erro leitura DICOM: {exc}')
            continue

        ok += 1
        manifest_rows.append(
            {
                'image_id': image_id,
                'subject_id': subject_id,
                'image_visit': row.get('image_visit'),
                'visit_for_dx': row.get('visit_for_dx'),
                'DIAGNOSIS': dx,
                'label': label,
                'match_method': row.get('match_method'),
                'dx_matched_viscode': row.get('dx_matched_viscode'),
                'dicom_series_dir': str(series_dir),
                'png_path': str(out_path.resolve()),
            },
        )

    if manifest_rows:
        mpath = args.manifest_csv
        mpath.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(manifest_rows).to_csv(mpath, index=False)

    print(f'\nPNG gerados com sucesso: {ok}')
    print(f'Linhas no merge antes do loop: {len(merged)} | Pastas I* no disco: {len(dicom_index)}')
    print(f'Estatísticas de merge: {args.merge_stats_txt.resolve()}')
    if ok > 0 and len(skipped) > ok:
        print(
            '\nNota: o Key MRI costuma listar a coorte inteira; faltas "sem pasta I*" indicam '
            'volumes não descarregados do LONI. Use --skip-missing-dicom noutra execução para '
            'só processar image_id presentes no disco (muito mais rápido).',
        )
    if skipped:
        print(f'Avisos / falhas ({len(skipped)}), primeiros 15:')
        for line in skipped[:15]:
            print(f'  - {line}')
        if len(skipped) > 15:
            print(f'  ... e mais {len(skipped) - 15}.')

    if ok == 0:
        raise RuntimeError(
            'Nenhuma imagem foi gerada. Confira --dicom-root / raízes MRI_1+MRI_2, '
            'dicom_roots no JSON ou --extra-dicom-root, e se image_id existem no disco.',
        )


if __name__ == '__main__':
    main()
