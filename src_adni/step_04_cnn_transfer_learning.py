from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from torchvision.models import ResNet18_Weights
from tqdm import tqdm


class SplitImageDataset(Dataset):
    # Dataset customizado para ler imagens e labels a partir do CSV de splits.
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
        # Retorna o total de amostras para o DataLoader.
        return len(self.split_df)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        # Carrega uma imagem, aplica transformações e converte a label para ID.
        row = self.split_df.iloc[index]
        image_path = Path(row['filepath'])
        label_text = str(row['label'])
        label_id = self.label_to_id[label_text]

        with Image.open(image_path) as image:
            rgb_image = image.convert('RGB')
            image_tensor = self.transform(rgb_image)

        return image_tensor, label_id


def parse_args() -> argparse.Namespace:
    # Define parâmetros de execução para facilitar experimentos e reprodução.
    parser = argparse.ArgumentParser(
        description='Treino de CNN com transfer learning (ResNet18) usando CSV de splits.',
    )
    parser.add_argument(
        '--split-csv',
        type=Path,
        default=Path('reports_adni/split_assignments.csv'),
        help='CSV gerado no passo 02.',
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('reports_adni/cnn_transfer_learning'),
        help='Pasta de saída para métricas, histórico e modelo.',
    )
    parser.add_argument(
        '--image-size',
        type=int,
        default=224,
        help='Tamanho da imagem para entrada da ResNet.',
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Tamanho do batch.',
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=3,
        help='Quantidade de épocas de treino.',
    )
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=1e-3,
        help='Taxa de aprendizado do otimizador Adam.',
    )
    parser.add_argument(
        '--freeze-backbone',
        action='store_true',
        help='Se ativado, treina apenas a camada final (mais rápido para começar).',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Semente para reprodutibilidade.',
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    # Fixa sementes principais para reduzir variação entre execuções.
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_split_dataframe(split_csv: Path) -> pd.DataFrame:
    # Valida o CSV de entrada e garante colunas mínimas do pipeline.
    if not split_csv.exists():
        raise FileNotFoundError(f'Arquivo de split não encontrado: {split_csv}')

    split_df = pd.read_csv(split_csv)
    required_columns = {'filepath', 'label', 'split'}
    missing_columns = required_columns.difference(split_df.columns)
    if missing_columns:
        raise ValueError(f'Colunas ausentes no CSV: {sorted(missing_columns)}')

    return split_df


def get_transforms(image_size: int) -> tuple[transforms.Compose, transforms.Compose]:
    # Define transformações de treino (com augment) e de avaliação (determinísticas).
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
    # Separa os splits e prepara mapeamento de classes para IDs.
    train_df = split_df[split_df['split'] == 'train'].copy()
    val_df = split_df[split_df['split'] == 'val'].copy()
    test_df = split_df[split_df['split'] == 'test'].copy()
    if train_df.empty or val_df.empty or test_df.empty:
        raise ValueError('Um dos splits está vazio; revise o CSV do passo 02.')

    labels = sorted(split_df['label'].unique().tolist())
    label_to_id = {label: idx for idx, label in enumerate(labels)}
    id_to_label = {idx: label for label, idx in label_to_id.items()}

    # Cria transforms e datasets para cada fase.
    train_transform, eval_transform = get_transforms(image_size=image_size)
    train_dataset = SplitImageDataset(train_df, label_to_id, train_transform)
    val_dataset = SplitImageDataset(val_df, label_to_id, eval_transform)
    test_dataset = SplitImageDataset(test_df, label_to_id, eval_transform)

    # DataLoaders para iteração eficiente em batches.
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, val_loader, test_loader, label_to_id, id_to_label


def compute_class_weights(train_loader: DataLoader, num_classes: int, device: torch.device) -> torch.Tensor:
    # Calcula pesos inversamente proporcionais para reduzir impacto do desbalanceamento.
    class_counts = torch.zeros(num_classes, dtype=torch.float32)
    for _, labels in train_loader:
        for label_id in labels:
            class_counts[int(label_id)] += 1.0

    class_weights = class_counts.sum() / torch.clamp(class_counts, min=1.0)
    class_weights = class_weights / class_weights.mean()
    return class_weights.to(device)


def build_model(num_classes: int, freeze_backbone: bool) -> nn.Module:
    # Carrega ResNet18 pré-treinada e ajusta a última camada para o número de classes.
    model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
    if freeze_backbone:
        # Congela backbone para acelerar o treino inicial e facilitar entendimento.
        for parameter in model.parameters():
            parameter.requires_grad = False

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    # Executa uma época de treino acumulando loss e acurácia.
    model.train()
    all_true: list[int] = []
    all_pred: list[int] = []
    running_loss = 0.0

    for images, labels in tqdm(dataloader, desc='Treinando', leave=False):
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
    # Avalia modelo sem atualizar pesos para validação ou teste.
    model.eval()
    all_true: list[int] = []
    all_pred: list[int] = []
    running_loss = 0.0

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc='Avaliando', leave=False):
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
    # Salva histórico, métricas e relatórios para comparação com outros modelos.
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

    print('\nArquivos gerados:')
    print(f'- {history_path.resolve()}')
    print(f'- {metrics_path.resolve()}')
    print(f'- {confusion_path.resolve()}')
    print(f'- {report_path.resolve()}')


def main() -> None:
    # Orquestra o treino completo de transfer learning e a geração de artefatos.
    args = parse_args()
    set_seed(args.seed)

    split_df = load_split_dataframe(args.split_csv)
    train_loader, val_loader, test_loader, label_to_id, id_to_label = build_dataloaders(
        split_df=split_df,
        image_size=args.image_size,
        batch_size=args.batch_size,
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Usando dispositivo: {device}')

    model = build_model(num_classes=len(label_to_id), freeze_backbone=args.freeze_backbone).to(device)
    class_weights = compute_class_weights(
        train_loader=train_loader,
        num_classes=len(label_to_id),
        device=device,
    )
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Treina apenas parâmetros com requires_grad=True (útil quando backbone está congelado).
    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = Adam(trainable_parameters, lr=args.learning_rate)

    best_val_accuracy = -1.0
    best_state_dict = None
    history_rows: list[dict[str, float]] = []

    for epoch in range(1, args.epochs + 1):
        print(f'\nÉpoca {epoch}/{args.epochs}')
        train_loss, train_accuracy = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )
        val_loss, val_accuracy, _, _ = evaluate(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
        )

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

        # Guarda o melhor estado por acurácia de validação.
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_state_dict = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    if best_state_dict is None:
        raise RuntimeError('Não foi possível salvar o melhor estado do modelo.')

    # Restaura melhor modelo para avaliar no teste.
    model.load_state_dict(best_state_dict)
    model_path = args.output_dir / 'best_model_resnet18.pth'
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(best_state_dict, model_path)

    test_loss, test_accuracy, test_true, test_pred = evaluate(
        model=model,
        dataloader=test_loader,
        criterion=criterion,
        device=device,
    )

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
        'freeze_backbone': bool(args.freeze_backbone),
        'best_val_accuracy': float(best_val_accuracy),
        'test_accuracy': float(test_accuracy),
        'test_loss': float(test_loss),
    }

    save_outputs(
        output_dir=args.output_dir,
        history_rows=history_rows,
        metrics=metrics,
        confusion_df=confusion_df,
        classification_report_dict=classification_report_dict,
    )

    print(f'\nModelo salvo em: {model_path.resolve()}')
    print(f'Best Val Accuracy: {best_val_accuracy:.4f}')
    print(f'Test Accuracy: {test_accuracy:.4f}')


if __name__ == '__main__':
    # Mantém execução direta via linha de comando.
    main()
