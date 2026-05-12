from __future__ import annotations

"""
Etapa 07b — ViT (variante experimental separada da Etapa 07).

Use `step_07_vit_transfer_learning.py` para reproduzir o experimento baseline já
reportado; use este arquivo quando quiser outra pasta de saída e hiperparâmetros
(p.ex. mais épocas) sem sobrescrever `reports_adni/vit_transfer_learning_grouped_binary`.

A implementação é a mesma da Etapa 07; apenas argumentos padrão e mensagens diferem.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from torchvision.models import ViT_B_16_Weights
from tqdm import tqdm


class SplitImageDataset(Dataset):
    # Dataset customizado para leitura das imagens a partir do CSV de split.
    def __init__(
        self,
        split_df: pd.DataFrame,
        label_to_id: dict[str, int],
        transform: transforms.Compose,
    ) -> None:
        self.split_df = split_df.reset_index(drop=True).copy()
        self.label_to_id = label_to_id
        self.transform = transform

    def __len__(self) -> int:
        # Retorna total de amostras do split atual.
        return len(self.split_df)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        # Carrega imagem e converte label textual para ID numérico.
        row = self.split_df.iloc[index]
        image_path = Path(row['filepath'])
        label_id = self.label_to_id[str(row['label'])]

        with Image.open(image_path) as image:
            rgb_image = image.convert('RGB')
            image_tensor = self.transform(rgb_image)

        return image_tensor, label_id


def parse_args() -> argparse.Namespace:
    # Parâmetros padrão distintos da Etapa 07 para comparar sem sobrescrever artefatos.
    parser = argparse.ArgumentParser(
        description='Etapa 07b: ViT (vit_b_16) — variante com pasta própria e mais épocas por padrão.',
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
        default=Path('reports_adni/vit_transfer_learning_grouped_binary_b'),
        help='Pasta de saída separada da Etapa 07 para comparação lado a lado.',
    )
    parser.add_argument(
        '--image-size',
        type=int,
        default=224,
        help='Tamanho da imagem de entrada para ViT.',
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=16,
        help='Tamanho do batch (ViT consome mais memória).',
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=8,
        help='Número de épocas (padrão 08b maior que o baseline 07 para explorar convergência).',
    )
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=3e-4,
        help='Taxa de aprendizado do AdamW.',
    )
    parser.add_argument(
        '--weight-decay',
        type=float,
        default=1e-4,
        help='Weight decay para regularização.',
    )
    parser.add_argument(
        '--freeze-backbone',
        action='store_true',
        help='Se ativo, treina apenas a cabeça de classificação.',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Semente para reprodutibilidade.',
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    # Fixa semente para reduzir variabilidade entre execuções.
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_split_dataframe(split_csv: Path) -> pd.DataFrame:
    # Valida o CSV de entrada para evitar falhas silenciosas.
    if not split_csv.exists():
        raise FileNotFoundError(f'Arquivo de split não encontrado: {split_csv}')

    split_df = pd.read_csv(split_csv)
    required_columns = {'filepath', 'label', 'split'}
    missing_columns = required_columns.difference(split_df.columns)
    if missing_columns:
        raise ValueError(f'Colunas ausentes no CSV: {sorted(missing_columns)}')
    return split_df


def get_transforms(image_size: int) -> tuple[transforms.Compose, transforms.Compose]:
    # Define pipeline de augmentação para treino e pipeline fixa para avaliação.
    train_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ],
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ],
    )
    return train_transform, eval_transform


def build_dataloaders(
    split_df: pd.DataFrame,
    image_size: int,
    batch_size: int,
) -> tuple[DataLoader, DataLoader, DataLoader, dict[str, int], dict[int, str]]:
    # Separa os três splits e cria mapeamento consistente de classes.
    train_df = split_df[split_df['split'] == 'train'].copy()
    val_df = split_df[split_df['split'] == 'val'].copy()
    test_df = split_df[split_df['split'] == 'test'].copy()
    if train_df.empty or val_df.empty or test_df.empty:
        raise ValueError('Um dos splits está vazio; revise o CSV.')

    labels = sorted(split_df['label'].unique().tolist())
    label_to_id = {label: idx for idx, label in enumerate(labels)}
    id_to_label = {idx: label for label, idx in label_to_id.items()}

    train_transform, eval_transform = get_transforms(image_size=image_size)
    train_dataset = SplitImageDataset(train_df, label_to_id, train_transform)
    val_dataset = SplitImageDataset(val_df, label_to_id, eval_transform)
    test_dataset = SplitImageDataset(test_df, label_to_id, eval_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, val_loader, test_loader, label_to_id, id_to_label


def compute_class_weights(train_loader: DataLoader, num_classes: int, device: torch.device) -> torch.Tensor:
    # Calcula pesos inversos por classe para lidar com desbalanceamento.
    class_counts = torch.zeros(num_classes, dtype=torch.float32)
    for _, labels in train_loader:
        for label_id in labels:
            class_counts[int(label_id)] += 1.0
    class_weights = class_counts.sum() / torch.clamp(class_counts, min=1.0)
    class_weights = class_weights / class_weights.mean()
    return class_weights.to(device)


def build_model(num_classes: int, freeze_backbone: bool) -> nn.Module:
    # Carrega ViT pré-treinado e ajusta a cabeça para o número de classes.
    model = models.vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
    if freeze_backbone:
        # Congela backbone para treino rápido da cabeça de classificação.
        for parameter in model.parameters():
            parameter.requires_grad = False

    in_features = model.heads.head.in_features
    model.heads.head = nn.Linear(in_features, num_classes)
    return model


def build_progress_bar(dataloader: DataLoader, description: str):
    # Mantém barra estável no PowerShell evitando quebra de linha por largura dinâmica.
    return tqdm(
        dataloader,
        desc=description,
        leave=False,
        dynamic_ncols=False,
        ncols=100,
        mininterval=1.0,
        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]',
    )


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    # Executa uma época de treino e retorna loss e acurácia.
    model.train()
    all_true: list[int] = []
    all_pred: list[int] = []
    running_loss = 0.0

    for images, labels in build_progress_bar(dataloader=dataloader, description='Treinando'):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += float(loss.item()) * images.size(0)
        predictions = torch.argmax(logits, dim=1)
        all_true.extend(labels.detach().cpu().numpy().tolist())
        all_pred.extend(predictions.detach().cpu().numpy().tolist())

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_accuracy = accuracy_score(all_true, all_pred)
    return epoch_loss, float(epoch_accuracy)


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float, list[int], list[int]]:
    # Avalia modelo em validação ou teste sem atualização de pesos.
    model.eval()
    all_true: list[int] = []
    all_pred: list[int] = []
    running_loss = 0.0

    with torch.no_grad():
        for images, labels in build_progress_bar(dataloader=dataloader, description='Avaliando'):
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)

            running_loss += float(loss.item()) * images.size(0)
            predictions = torch.argmax(logits, dim=1)
            all_true.extend(labels.detach().cpu().numpy().tolist())
            all_pred.extend(predictions.detach().cpu().numpy().tolist())

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_accuracy = accuracy_score(all_true, all_pred)
    return epoch_loss, float(epoch_accuracy), all_true, all_pred


def save_outputs(
    output_dir: Path,
    history_rows: list[dict[str, float]],
    metrics: dict[str, float],
    confusion_df: pd.DataFrame,
    classification_report_dict: dict[str, object],
) -> None:
    # Salva artefatos de treino para comparação com outros modelos.
    output_dir.mkdir(parents=True, exist_ok=True)
    history_path = output_dir / 'history.csv'
    metrics_path = output_dir / 'metrics.json'
    confusion_path = output_dir / 'confusion_matrix.csv'
    report_path = output_dir / 'classification_report_test.json'

    pd.DataFrame(history_rows).to_csv(history_path, index=False)
    confusion_df.to_csv(confusion_path, index=True)
    with metrics_path.open('w', encoding='utf-8') as metrics_file:
        json.dump(metrics, metrics_file, indent=2, ensure_ascii=False)
    with report_path.open('w', encoding='utf-8') as report_file:
        json.dump(classification_report_dict, report_file, indent=2, ensure_ascii=False)

    print('\nArquivos gerados (Etapa 07b):')
    print(f'- {history_path.resolve()}')
    print(f'- {metrics_path.resolve()}')
    print(f'- {confusion_path.resolve()}')
    print(f'- {report_path.resolve()}')


def main() -> None:
    # Orquestra treino completo do ViT e avaliação final no conjunto de teste.
    args = parse_args()
    set_seed(args.seed)

    split_df = load_split_dataframe(args.split_csv)
    train_loader, val_loader, test_loader, label_to_id, id_to_label = build_dataloaders(
        split_df=split_df,
        image_size=args.image_size,
        batch_size=args.batch_size,
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[07b] Dispositivo: {device}')

    model = build_model(num_classes=len(label_to_id), freeze_backbone=args.freeze_backbone).to(device)
    class_weights = compute_class_weights(train_loader, len(label_to_id), device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = AdamW(
        trainable_parameters,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    best_val_accuracy = -1.0
    best_state_dict = None
    history_rows: list[dict[str, float]] = []

    for epoch in range(1, args.epochs + 1):
        print(f'\n[07b] Época {epoch}/{args.epochs}')
        train_loss, train_accuracy = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_accuracy, _, _ = evaluate(model, val_loader, criterion, device)

        history_rows.append(
            {
                'epoch': float(epoch),
                'train_loss': float(train_loss),
                'train_accuracy': float(train_accuracy),
                'val_loss': float(val_loss),
                'val_accuracy': float(val_accuracy),
            },
        )
        print(
            f'train_loss={train_loss:.4f} | train_acc={train_accuracy:.4f} | '
            f'val_loss={val_loss:.4f} | val_acc={val_accuracy:.4f}',
        )

        if val_accuracy > best_val_accuracy:
            # Guarda melhor estado com base na validação.
            best_val_accuracy = val_accuracy
            best_state_dict = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    if best_state_dict is None:
        raise RuntimeError('Não foi possível salvar o melhor estado do modelo ViT (07b).')

    model.load_state_dict(best_state_dict)
    model_path = args.output_dir / 'best_model_vit_b_16.pth'
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(best_state_dict, model_path)

    test_loss, test_accuracy, test_true, test_pred = evaluate(model, test_loader, criterion, device)
    target_names = [id_to_label[idx] for idx in range(len(id_to_label))]
    classification_report_dict = classification_report(
        test_true,
        test_pred,
        target_names=target_names,
        output_dict=True,
        zero_division=0,
    )
    confusion = confusion_matrix(test_true, test_pred)
    confusion_df = pd.DataFrame(confusion, index=target_names, columns=target_names)

    metrics = {
        'device': str(device),
        'epochs': int(args.epochs),
        'batch_size': int(args.batch_size),
        'image_size': int(args.image_size),
        'learning_rate': float(args.learning_rate),
        'weight_decay': float(args.weight_decay),
        'freeze_backbone': bool(args.freeze_backbone),
        'best_val_accuracy': float(best_val_accuracy),
        'test_accuracy': float(test_accuracy),
        'test_loss': float(test_loss),
    }
    save_outputs(args.output_dir, history_rows, metrics, confusion_df, classification_report_dict)

    print(f'\n[07b] Modelo salvo em: {model_path.resolve()}')
    print(f'[07b] Best Val Accuracy: {best_val_accuracy:.4f}')
    print(f'[07b] Test Accuracy: {test_accuracy:.4f}')


if __name__ == '__main__':
    # Mantém execução direta por linha de comando.
    main()
