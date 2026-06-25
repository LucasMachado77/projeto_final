from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Treino ADNI 2D com MONAI ResNet18 usando CSV de split por paciente.',
    )
    parser.add_argument(
        '--split-csv',
        type=Path,
        default=Path('reports_adni/split_assignments_grouped_binary.csv'),
        help='CSV com colunas filepath, label e split.',
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('reports_adni/monai_2d_resnet18_grouped_binary'),
        help='Pasta de saida para modelo, historico e metricas.',
    )
    parser.add_argument('--image-size', type=int, default=224, help='Tamanho HxW da imagem.')
    parser.add_argument('--batch-size', type=int, default=32, help='Tamanho do batch.')
    parser.add_argument('--epochs', type=int, default=20, help='Numero maximo de epocas.')
    parser.add_argument('--learning-rate', type=float, default=1e-3, help='Taxa de aprendizado inicial.')
    parser.add_argument('--weight-decay', type=float, default=1e-4, help='Weight decay do Adam.')
    parser.add_argument('--num-workers', type=int, default=0, help='Workers do DataLoader.')
    parser.add_argument(
        '--early-stopping-patience',
        type=int,
        default=8,
        help='Epocas sem melhoria antes de parar (0 desativa).',
    )
    parser.add_argument(
        '--best-by',
        choices=('val_loss', 'val_accuracy'),
        default='val_loss',
        help='Metrica usada para salvar o melhor checkpoint.',
    )
    parser.add_argument('--min-delta', type=float, default=1e-4, help='Melhoria minima para contar progresso.')
    parser.add_argument('--lr-scheduler-patience', type=int, default=2, help='Paciencia do ReduceLROnPlateau.')
    parser.add_argument('--lr-factor', type=float, default=0.5, help='Fator de reducao de LR.')
    parser.add_argument('--lr-min', type=float, default=1e-6, help='LR minimo.')
    parser.add_argument('--grad-clip-norm', type=float, default=1.0, help='clip_grad_norm_ global (0 desativa).')
    parser.add_argument('--seed', type=int, default=42, help='Semente para reproducibilidade.')
    parser.add_argument(
        '--skip-missing-paths',
        action='store_true',
        help='Remove linhas cujo filepath nao existe, em vez de falhar.',
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
        '--dry-run',
        action='store_true',
        help='Valida transforms/DataLoader e sai sem treinar.',
    )
    return parser.parse_args()


def import_monai(output_dir: Path):
    cache_dir = output_dir / '.matplotlib_cache'
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault('MPLCONFIGDIR', str(cache_dir.resolve()))

    try:
        from monai.data import DataLoader, Dataset
        from monai.networks.nets import resnet18
        from monai.transforms import Compose, LoadImaged, RandFlipd, RandRotated, Resized, ScaleIntensityd, ToTensord
    except ModuleNotFoundError as exc:
        if exc.name == 'monai':
            raise ModuleNotFoundError('MONAI nao esta instalado. Rode: pip install -r requirements.txt') from exc
        raise

    return Compose, LoadImaged, RandFlipd, RandRotated, Resized, ScaleIntensityd, ToTensord, Dataset, DataLoader, resnet18


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def load_split_dataframe(
    split_csv: Path,
    skip_missing_paths: bool,
    path_prefix_from: Path | None,
    path_prefix_to: Path | None,
) -> pd.DataFrame:
    if not split_csv.exists():
        raise FileNotFoundError(f'CSV de split nao encontrado: {split_csv}')
    split_df = pd.read_csv(split_csv)
    required_columns = {'filepath', 'label', 'split'}
    missing_columns = required_columns.difference(split_df.columns)
    if missing_columns:
        raise ValueError(f'Colunas ausentes no CSV: {sorted(missing_columns)}')

    split_df = split_df.copy()
    split_df['filepath'] = split_df['filepath'].apply(
        lambda value: str(remap_path(Path(str(value)), path_prefix_from, path_prefix_to)),
    )
    split_df['path_exists'] = split_df['filepath'].apply(lambda value: Path(str(value)).is_file())
    missing_df = split_df[~split_df['path_exists']]
    if not missing_df.empty and not skip_missing_paths:
        examples = '\n'.join(str(path) for path in missing_df['filepath'].head(5))
        raise FileNotFoundError(
            f'{len(missing_df)} imagens do CSV nao existem no disco. Exemplos:\n{examples}\n'
            'Use --skip-missing-paths apenas para smoke tests.',
        )
    if skip_missing_paths:
        split_df = split_df[split_df['path_exists']].copy()

    return split_df.drop(columns=['path_exists'])


def build_records(split_df: pd.DataFrame, label_to_id: dict[str, int]) -> list[dict[str, object]]:
    return [
        {
            'image': str(row.filepath),
            'label': int(label_to_id[str(row.label)]),
        }
        for row in split_df.itertuples(index=False)
    ]


class EnsureSingleChannelD:
    def __init__(self, keys: tuple[str, ...]) -> None:
        self.keys = keys

    def __call__(self, data: dict[str, object]) -> dict[str, object]:
        result = dict(data)
        for key in self.keys:
            image = result[key]
            if image.shape[0] == 1:
                continue
            if isinstance(image, torch.Tensor):
                result[key] = image[:3].mean(dim=0, keepdim=True)
            else:
                result[key] = np.asarray(image[:3]).mean(axis=0, keepdims=True)
        return result


def build_transforms(image_size: int, output_dir: Path):
    Compose, LoadImaged, RandFlipd, RandRotated, Resized, ScaleIntensityd, ToTensord, *_ = import_monai(output_dir)
    train_transform = Compose(
        [
            LoadImaged(keys='image', image_only=True, ensure_channel_first=True),
            EnsureSingleChannelD(keys=('image',)),
            ScaleIntensityd(keys='image', minv=0.0, maxv=1.0),
            Resized(keys='image', spatial_size=(image_size, image_size), mode='bilinear'),
            RandFlipd(keys='image', prob=0.5, spatial_axis=1),
            RandRotated(keys='image', range_x=np.deg2rad(10), prob=0.5, keep_size=True, mode='bilinear'),
            ToTensord(keys=('image', 'label')),
        ],
    )
    eval_transform = Compose(
        [
            LoadImaged(keys='image', image_only=True, ensure_channel_first=True),
            EnsureSingleChannelD(keys=('image',)),
            ScaleIntensityd(keys='image', minv=0.0, maxv=1.0),
            Resized(keys='image', spatial_size=(image_size, image_size), mode='bilinear'),
            ToTensord(keys=('image', 'label')),
        ],
    )
    return train_transform, eval_transform


def build_dataloaders(
    split_df: pd.DataFrame,
    image_size: int,
    batch_size: int,
    num_workers: int,
    output_dir: Path,
):
    *_, Dataset, DataLoader, _resnet18 = import_monai(output_dir)

    train_df = split_df[split_df['split'] == 'train'].copy()
    val_df = split_df[split_df['split'] == 'val'].copy()
    test_df = split_df[split_df['split'] == 'test'].copy()
    if train_df.empty or val_df.empty or test_df.empty:
        raise ValueError('Um dos splits esta vazio; revise o CSV de entrada.')

    labels = sorted(split_df['label'].astype(str).unique().tolist())
    label_to_id = {label: idx for idx, label in enumerate(labels)}
    id_to_label = {idx: label for label, idx in label_to_id.items()}

    train_transform, eval_transform = build_transforms(image_size=image_size, output_dir=output_dir)
    train_dataset = Dataset(data=build_records(train_df, label_to_id), transform=train_transform)
    val_dataset = Dataset(data=build_records(val_df, label_to_id), transform=eval_transform)
    test_dataset = Dataset(data=build_records(test_df, label_to_id), transform=eval_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, test_loader, label_to_id, id_to_label


def compute_class_weights(split_df: pd.DataFrame, label_to_id: dict[str, int], device: torch.device) -> torch.Tensor:
    train_labels = split_df[split_df['split'] == 'train']['label'].astype(str)
    counts = torch.zeros(len(label_to_id), dtype=torch.float32)
    for label in train_labels:
        counts[label_to_id[label]] += 1.0
    weights = counts.sum() / torch.clamp(counts, min=1.0)
    weights = weights / weights.mean()
    return weights.to(device)


def build_model(num_classes: int, output_dir: Path) -> nn.Module:
    *_, resnet18 = import_monai(output_dir)
    return resnet18(spatial_dims=2, n_input_channels=1, num_classes=num_classes)


def run_epoch(
    model: nn.Module,
    dataloader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    grad_clip_norm: float = 0.0,
) -> tuple[float, float, list[int], list[int]]:
    is_train = optimizer is not None
    model.train(mode=is_train)
    running_loss = 0.0
    all_true: list[int] = []
    all_pred: list[int] = []

    context = torch.enable_grad() if is_train else torch.no_grad()
    desc = 'Treinando' if is_train else 'Avaliando'
    with context:
        for batch in tqdm(dataloader, desc=desc, leave=False):
            images = batch['image'].to(device)
            labels = batch['label'].to(device).long()

            if is_train:
                optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            if is_train:
                loss.backward()
                if grad_clip_norm > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
                optimizer.step()

            running_loss += float(loss.item()) * images.size(0)
            predictions = torch.argmax(logits, dim=1)
            all_true.extend(labels.detach().cpu().numpy().tolist())
            all_pred.extend(predictions.detach().cpu().numpy().tolist())

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_accuracy = accuracy_score(all_true, all_pred)
    return epoch_loss, float(epoch_accuracy), all_true, all_pred


def is_improved(best_by: str, val_loss: float, val_accuracy: float, best_loss: float, best_acc: float, min_delta: float) -> bool:
    if best_by == 'val_loss':
        return val_loss < best_loss - min_delta
    return val_accuracy > best_acc + min_delta


def save_outputs(
    output_dir: Path,
    history_rows: list[dict[str, float]],
    metrics: dict[str, object],
    confusion_df: pd.DataFrame,
    classification_report_dict: dict[str, object],
    label_to_id: dict[str, int],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history_rows).to_csv(output_dir / 'history.csv', index=False)
    confusion_df.to_csv(output_dir / 'confusion_matrix.csv', index=True)
    (output_dir / 'metrics.json').write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    (output_dir / 'classification_report_test.json').write_text(
        json.dumps(classification_report_dict, indent=2, ensure_ascii=False) + '\n',
        encoding='utf-8',
    )
    (output_dir / 'class_mapping.json').write_text(json.dumps(label_to_id, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    split_df = load_split_dataframe(
        args.split_csv,
        skip_missing_paths=args.skip_missing_paths,
        path_prefix_from=args.path_prefix_from,
        path_prefix_to=args.path_prefix_to,
    )
    train_loader, val_loader, test_loader, label_to_id, id_to_label = build_dataloaders(
        split_df=split_df,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        output_dir=args.output_dir,
    )

    if args.dry_run:
        batch = next(iter(train_loader))
        images = batch['image']
        labels = batch['label']
        print('\nDry-run MONAI concluido:')
        print(f'- batch image shape: {list(images.shape)}')
        print(f'- batch label shape: {list(labels.shape)}')
        print(f'- intensidade: {float(images.min()):.4f} a {float(images.max()):.4f}')
        print(f'- classes: {label_to_id}')
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Usando dispositivo: {device}')
    model = build_model(num_classes=len(label_to_id), output_dir=args.output_dir).to(device)

    class_weights = compute_class_weights(split_df=split_df, label_to_id=label_to_id, device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=args.lr_factor,
        patience=args.lr_scheduler_patience,
        min_lr=args.lr_min,
    )

    best_state_dict = None
    best_epoch = 0
    best_val_loss = float('inf')
    best_val_accuracy = -1.0
    stale_epochs = 0
    history_rows: list[dict[str, float]] = []

    for epoch in range(1, args.epochs + 1):
        print(f'\nEpoca {epoch}/{args.epochs} | lr={optimizer.param_groups[0]["lr"]:.2e}')
        train_loss, train_accuracy, _, _ = run_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            device=device,
            optimizer=optimizer,
            grad_clip_norm=args.grad_clip_norm,
        )
        val_loss, val_accuracy, _, _ = run_epoch(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
        )
        scheduler.step(val_loss)

        improved = is_improved(
            best_by=args.best_by,
            val_loss=val_loss,
            val_accuracy=val_accuracy,
            best_loss=best_val_loss,
            best_acc=best_val_accuracy,
            min_delta=args.min_delta,
        )
        if improved:
            stale_epochs = 0
            best_epoch = epoch
            best_val_loss = val_loss
            best_val_accuracy = val_accuracy
            best_state_dict = {key: value.detach().cpu() for key, value in model.state_dict().items()}
        else:
            stale_epochs += 1

        history_rows.append(
            {
                'epoch': epoch,
                'train_loss': train_loss,
                'train_accuracy': train_accuracy,
                'val_loss': val_loss,
                'val_accuracy': val_accuracy,
                'lr': float(optimizer.param_groups[0]['lr']),
            },
        )
        print(
            f'train_loss={train_loss:.4f} | train_acc={train_accuracy:.4f} | '
            f'val_loss={val_loss:.4f} | val_acc={val_accuracy:.4f} | '
            f'best_epoch={best_epoch} | sem melhoria={stale_epochs}'
        )

        if args.early_stopping_patience > 0 and stale_epochs >= args.early_stopping_patience:
            print(f'\nEarly stopping: sem melhoria em {args.early_stopping_patience} epocas.')
            break

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    test_loss, test_accuracy, test_true, test_pred = run_epoch(
        model=model,
        dataloader=test_loader,
        criterion=criterion,
        device=device,
    )

    model_path = args.output_dir / 'best_model_monai_resnet18.pth'
    torch.save(model.state_dict(), model_path)

    labels_order = [id_to_label[index] for index in sorted(id_to_label)]
    confusion = confusion_matrix(test_true, test_pred, labels=sorted(id_to_label))
    confusion_df = pd.DataFrame(confusion, index=labels_order, columns=labels_order)
    report = classification_report(
        test_true,
        test_pred,
        labels=sorted(id_to_label),
        target_names=labels_order,
        output_dict=True,
        zero_division=0,
    )
    metrics = {
        'framework': 'MONAI',
        'model': 'monai.networks.nets.resnet18',
        'pretrained': False,
        'input_channels': 1,
        'device': str(device),
        'epochs_requested': args.epochs,
        'epochs_ran': len(history_rows),
        'batch_size': args.batch_size,
        'image_size': args.image_size,
        'learning_rate': args.learning_rate,
        'weight_decay': args.weight_decay,
        'best_by': args.best_by,
        'best_epoch': best_epoch,
        'best_val_loss': best_val_loss,
        'best_val_accuracy': best_val_accuracy,
        'test_loss': test_loss,
        'test_accuracy': test_accuracy,
        'model_path': str(model_path),
    }
    save_outputs(
        output_dir=args.output_dir,
        history_rows=history_rows,
        metrics=metrics,
        confusion_df=confusion_df,
        classification_report_dict=report,
        label_to_id=label_to_id,
    )

    print('\nTreino MONAI 2D concluido:')
    print(f'- Modelo: {model_path.resolve()}')
    print(f'- Best epoch: {best_epoch}')
    print(f'- Test Accuracy: {test_accuracy:.4f}')


if __name__ == '__main__':
    main()
