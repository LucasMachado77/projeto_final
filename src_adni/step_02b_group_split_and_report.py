from __future__ import annotations

import argparse
import random
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
    parser.add_argument(
        '--balance-classes',
        choices=('none', 'downsample'),
        default='none',
        help=(
            'downsample: corta a classe majoritária aleatóriamente até ter o mesmo nº de imagens '
            'que a minoria (~50/50 global antes do split). none: mantém todas as imagens.'
        ),
    )
    parser.add_argument(
        '--max-images-per-subject',
        type=int,
        default=0,
        help=(
            'Se > 0, antes do balanceamento mantém no máximo N imagens por sujeito (amostragem aleatória). '
            'Ajuda quando poucos pacientes concentrarem muitas séries.'
        ),
    )
    parser.add_argument(
        '--duplicate-subject-policy',
        choices=('longitudinal', 'error', 'exclude', 'majority', 'any_demented'),
        default='longitudinal',
        help=(
            'ADNI longitudinal: o mesmo subject_id pode ter imagens Non Demented e Demented (evolução). '
            'longitudinal (padrão): split por paciente sem vazamento; cada imagem mantém a label da visita; '
            'estratificação usa se o sujeito teve alguma vez exame Demented. '
            'error: falha se houver labels mistas. exclude/majority/any_demented: simplificam para uma classe '
            'por paciente (descartam exames — só use se o desenho do estudo exigir).'
        ),
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


def cap_images_per_subject(dataset_df: pd.DataFrame, max_per_subject: int, seed: int) -> pd.DataFrame:
    # Limita quantas fatias por paciente entram, para não dominar o treino com poucos sujeitos.
    rng = random.Random(seed)
    chunks: list[pd.DataFrame] = []
    for _, group in dataset_df.groupby('subject_id', sort=False):
        if len(group) <= max_per_subject:
            chunks.append(group)
            continue
        positions = rng.sample(range(len(group)), k=max_per_subject)
        chunks.append(group.iloc[sorted(positions)])
    out = pd.concat(chunks, ignore_index=True)
    before, after = len(dataset_df), len(out)
    print(f'--max-images-per-subject={max_per_subject}: {before} -> {after} imagens.')
    return out


def balance_classes_downsample(dataset_df: pd.DataFrame, seed: int) -> pd.DataFrame:
    # Iguala contagens por classe (imagens): amostra sem reposição na maioria até n = min classe.
    counts = dataset_df['label'].value_counts()
    if len(counts) < 2:
        print('Aviso: só uma classe; balance-classes=downsample ignorado.')
        return dataset_df
    target_n = int(counts.min())
    parts: list[pd.DataFrame] = []
    for label in counts.index:
        g = dataset_df[dataset_df['label'] == label]
        if len(g) <= target_n:
            parts.append(g)
        else:
            parts.append(g.sample(n=target_n, random_state=seed, replace=False))
    out = pd.concat(parts, ignore_index=True)
    new_counts = out['label'].value_counts()
    print(
        'Balanceamento (downsample para a minoria): '
        f'{counts.to_dict()} -> {new_counts.to_dict()} | total {len(out)} imagens.',
    )
    return out


def _conflicting_subject_ids(dataset_df: pd.DataFrame) -> list[str]:
    # Sujeitos que aparecem com mais do que um valor em `label` (pastas diferentes).
    nunique = dataset_df.groupby('subject_id')['label'].nunique()
    return nunique[nunique > 1].index.tolist()


def resolve_duplicate_subject_labels(dataset_df: pd.DataFrame, policy: str) -> pd.DataFrame:
    # Para políticas que descartam ou colapsam labels; longitudinal não altera linhas (split trata no grupo).
    if policy == 'longitudinal':
        return dataset_df

    conflict_ids = _conflicting_subject_ids(dataset_df)
    if not conflict_ids:
        return dataset_df

    if policy == 'error':
        sample = dataset_df[dataset_df['subject_id'].isin(conflict_ids)].sort_values(
            ['subject_id', 'label'],
        )
        preview = sample.head(25).to_string(index=False)
        raise ValueError(
            'Há sujeitos com mais de uma label (frequentemente ADNI: evolução CN/MCI → AD). '
            f'Total com conflito: {len(conflict_ids)}. '
            'Use --duplicate-subject-policy longitudinal (recomendado ADNI) | exclude | majority | any_demented.\n'
            f'Pré-visualização (até 25 linhas):\n{preview}',
        )

    if policy == 'exclude':
        before_n = len(dataset_df)
        out = dataset_df[~dataset_df['subject_id'].isin(conflict_ids)].copy()
        print(
            f'--duplicate-subject-policy=exclude: removidas {before_n - len(out)} imagens '
            f'({len(conflict_ids)} sujeitos com labels mistas).',
        )
        if out.empty:
            raise ValueError('Após exclude não restaram imagens; reduza restrições ou mude a política.')
        return out

    def assign_majority(labels: pd.Series) -> str:
        # Label mais frequente por sujeito; empate favorece Demented.
        vc = labels.value_counts()
        best_count = int(vc.max())
        tied = [str(x) for x in vc.index if int(vc[x]) == best_count]
        if len(tied) > 1:
            return 'Demented' if 'Demented' in tied else sorted(tied)[0]
        return str(vc.index[0])

    def assign_any_demented(labels: pd.Series) -> str:
        # Qualquer exame Demented: sujeito fica na classe Demented; remove-se o resto das linhas desse sujeito.
        if (labels == 'Demented').any():
            return 'Demented'
        return 'Non Demented'

    if policy == 'majority':
        resolver = assign_majority
    elif policy == 'any_demented':
        resolver = assign_any_demented
    else:
        raise ValueError(f'Política desconhecida: {policy}')

    resolved_series = dataset_df.groupby('subject_id', sort=False)['label'].agg(resolver)
    merged = dataset_df.copy()
    merged['_resolved_label'] = merged['subject_id'].map(resolved_series)
    before_n = len(merged)
    out = merged[merged['label'] == merged['_resolved_label']].drop(columns=['_resolved_label']).copy()
    dropped = before_n - len(out)
    print(
        f'--duplicate-subject-policy={policy}: {len(conflict_ids)} sujeitos com labels mistas; '
        f'removidas {dropped} imagens cuja pasta não coincide com a label resolvida por sujeito.',
    )
    if out.empty:
        raise ValueError('Após resolver duplicados não restaram imagens.')
    return out


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
    *,
    longitudinal: bool,
) -> pd.DataFrame:
    # longitudinal=True: um split (train/val/test) por sujeito; label por exame inalterada (progressão natural).
    if longitudinal:

        def subject_stratify_label(labels: pd.Series) -> str:
            # Mantém proporção semelhante de sujeitos “com algum exame Demented” entre conjuntos.
            return 'Demented' if (labels == 'Demented').any() else 'Non Demented'

        strat = dataset_df.groupby('subject_id', sort=False)['label'].agg(subject_stratify_label)
        subject_df = strat.rename('stratify_label').reset_index()
        stratify_first = get_stratify_labels_or_none(subject_df['stratify_label'], 'train_vs_temp')
        train_subjects, temp_subjects = train_test_split(
            subject_df,
            test_size=(1.0 - train_size),
            random_state=seed,
            stratify=stratify_first,
        )
        val_ratio_inside_temp = val_size / (val_size + test_size)
        stratify_second = get_stratify_labels_or_none(temp_subjects['stratify_label'], 'val_vs_test')
        val_subjects, test_subjects = train_test_split(
            temp_subjects,
            test_size=(1.0 - val_ratio_inside_temp),
            random_state=seed,
            stratify=stratify_second,
        )
        train_subjects = train_subjects.copy()
        val_subjects = val_subjects.copy()
        test_subjects = test_subjects.copy()
        train_subjects['split'] = 'train'
        val_subjects['split'] = 'val'
        test_subjects['split'] = 'test'
        subject_split_df = pd.concat([train_subjects, val_subjects, test_subjects], ignore_index=True)
        split_df = dataset_df.merge(
            subject_split_df[['subject_id', 'split']],
            on='subject_id',
            how='inner',
        )
        n_mixed = len(_conflicting_subject_ids(dataset_df))
        if n_mixed:
            print(
                f'Modo longitudinal: {n_mixed} sujeitos com exames em ambas as classes; '
                'todas as imagens do mesmo paciente partilham split; label = pasta de cada exame.',
            )
        return split_df

    # Um único rótulo clínico por paciente (após resolve_*).
    subject_df = dataset_df[['subject_id', 'label']].drop_duplicates()
    duplicated_subject_labels = subject_df.duplicated(subset=['subject_id'], keep=False)
    if duplicated_subject_labels.any():
        raise ValueError(
            'Há sujeitos com mais de uma label; use --duplicate-subject-policy longitudinal '
            'ou aplique exclude/majority/any_demented antes do split.',
        )
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
    # Resolver o mesmo paciente em duas pastas antes de caps/balance (ADNI longitudinal).
    dataset_df = resolve_duplicate_subject_labels(dataset_df, args.duplicate_subject_policy)
    if args.max_images_per_subject > 0:
        dataset_df = cap_images_per_subject(
            dataset_df,
            args.max_images_per_subject,
            args.seed,
        )
    if args.balance_classes == 'downsample':
        dataset_df = balance_classes_downsample(dataset_df, args.seed)

    split_df = split_subjects(
        dataset_df=dataset_df,
        train_size=args.train_size,
        val_size=args.val_size,
        test_size=args.test_size,
        seed=args.seed,
        longitudinal=args.duplicate_subject_policy == 'longitudinal',
    )
    verify_no_subject_leakage(split_df)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    split_df.to_csv(output_csv, index=False)

    print_report(split_df)
    print(f'\nCSV gerado em: {output_csv.resolve()}')


if __name__ == '__main__':
    # Mantém execução direta em linha de comando.
    main()
