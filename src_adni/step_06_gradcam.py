from __future__ import annotations

import argparse
import json
from pathlib import Path

from matplotlib import colormaps
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import nn
from torchvision import models, transforms


def parse_args() -> argparse.Namespace:
    # Define parâmetros de entrada para gerar visualizações Grad-CAM.
    parser = argparse.ArgumentParser(
        description='Gera Grad-CAM para amostras do conjunto de teste.',
    )
    parser.add_argument(
        '--split-csv',
        type=Path,
        default=Path('reports_adni/split_assignments_grouped_binary.csv'),
        help='CSV com colunas filepath, label e split.',
    )
    parser.add_argument(
        '--model-path',
        type=Path,
        default=Path('reports_adni/cnn_transfer_learning_e5_grouped_binary/best_model_resnet18.pth'),
        help='Caminho do modelo treinado (state_dict).',
    )
    parser.add_argument(
        '--image-size',
        type=int,
        default=224,
        help='Tamanho da imagem usado no treino da CNN.',
    )
    parser.add_argument(
        '--samples-per-class',
        type=int,
        default=2,
        help='Quantidade de exemplos por classe para gerar Grad-CAM.',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Semente para seleção reprodutível de amostras.',
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('reports_adni/gradcam_grouped_binary'),
        help='Pasta de saída das imagens e metadados.',
    )
    return parser.parse_args()


def load_split_dataframe(split_csv: Path) -> pd.DataFrame:
    # Carrega e valida estrutura mínima do CSV de split.
    if not split_csv.exists():
        raise FileNotFoundError(f'CSV não encontrado: {split_csv}')
    split_df = pd.read_csv(split_csv)
    required_columns = {'filepath', 'label', 'split'}
    missing_columns = required_columns.difference(split_df.columns)
    if missing_columns:
        raise ValueError(f'Colunas ausentes no CSV: {sorted(missing_columns)}')
    return split_df


def build_model(num_classes: int, model_path: Path, device: torch.device) -> nn.Module:
    # Reconstrói a ResNet18 com a mesma cabeça do treino e carrega os pesos.
    model = models.resnet18(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def get_eval_transform(image_size: int) -> transforms.Compose:
    # Mantém o mesmo pré-processamento de avaliação usado no treino.
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ],
    )


def normalize_heatmap(heatmap: np.ndarray) -> np.ndarray:
    # Normaliza para [0, 1] com segurança numérica.
    min_value = float(heatmap.min())
    max_value = float(heatmap.max())
    denom = max(max_value - min_value, 1e-8)
    return (heatmap - min_value) / denom


def build_overlay_image(original_rgb: np.ndarray, heatmap: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    # Converte heatmap em cor (jet) e mistura com imagem original para visualização.
    heatmap_colored = colormaps['jet'](heatmap)[:, :, :3]
    heatmap_uint8 = (heatmap_colored * 255.0).astype(np.uint8)
    overlay = ((1.0 - alpha) * original_rgb.astype(np.float32) + alpha * heatmap_uint8.astype(np.float32))
    return np.clip(overlay, 0, 255).astype(np.uint8)


def compute_gradcam(
    model: nn.Module,
    image_tensor: torch.Tensor,
    target_layer: nn.Module,
    class_index: int,
) -> np.ndarray:
    # Calcula Grad-CAM com hooks na última camada convolucional.
    activations: list[torch.Tensor] = []
    gradients: list[torch.Tensor] = []

    def forward_hook(_module: nn.Module, _input: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        # Guarda ativações para cálculo posterior do mapa.
        activations.append(output.detach())

    def backward_hook(
        _module: nn.Module,
        grad_input: tuple[torch.Tensor, ...],
        grad_output: tuple[torch.Tensor, ...],
    ) -> None:
        # Guarda gradientes da saída da camada alvo.
        del grad_input
        gradients.append(grad_output[0].detach())

    forward_handle = target_layer.register_forward_hook(forward_hook)
    backward_handle = target_layer.register_full_backward_hook(backward_hook)

    try:
        model.zero_grad(set_to_none=True)
        logits = model(image_tensor)
        target_score = logits[:, class_index].sum()
        target_score.backward()

        activation_map = activations[0][0]
        gradient_map = gradients[0][0]
        weights = gradient_map.mean(dim=(1, 2), keepdim=True)
        cam = (weights * activation_map).sum(dim=0)
        cam = torch.relu(cam)
        cam_np = cam.cpu().numpy()
        return normalize_heatmap(cam_np)
    finally:
        # Remove hooks para evitar acúmulo entre iterações.
        forward_handle.remove()
        backward_handle.remove()


def main() -> None:
    # Orquestra seleção de imagens, inferência, Grad-CAM e persistência dos resultados.
    args = parse_args()
    np.random.seed(args.seed)

    split_df = load_split_dataframe(args.split_csv)
    test_df = split_df[split_df['split'] == 'test'].copy()
    if test_df.empty:
        raise ValueError('Split de teste vazio no CSV informado.')

    classes = sorted(test_df['label'].unique().tolist())
    class_to_index = {class_name: idx for idx, class_name in enumerate(classes)}
    index_to_class = {idx: class_name for class_name, idx in class_to_index.items()}

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if not args.model_path.exists():
        raise FileNotFoundError(f'Modelo não encontrado: {args.model_path}')
    model = build_model(
        num_classes=len(classes),
        model_path=args.model_path,
        device=device,
    )

    # Usa última camada convolucional do bloco final da ResNet18.
    target_layer = model.layer4[1].conv2
    eval_transform = get_eval_transform(args.image_size)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata_rows: list[dict[str, object]] = []
    for class_name in classes:
        # Seleciona amostras por classe para manter explicações equilibradas.
        class_df = test_df[test_df['label'] == class_name].copy()
        sample_count = min(args.samples_per_class, len(class_df))
        sampled_df = class_df.sample(n=sample_count, random_state=args.seed)

        for row_idx, row in enumerate(sampled_df.itertuples(index=False), start=1):
            image_path = Path(str(row.filepath))
            if not image_path.exists():
                continue

            # Carrega imagem original para salvar visualização final.
            with Image.open(image_path) as image:
                rgb_image = image.convert('RGB')
                resized_rgb = rgb_image.resize((args.image_size, args.image_size))
                original_np = np.array(resized_rgb, dtype=np.uint8)

            # Prepara tensor normalizado para inferência.
            image_tensor = eval_transform(resized_rgb).unsqueeze(0).to(device)

            # Prediz classe e calcula Grad-CAM para classe prevista.
            with torch.no_grad():
                logits = model(image_tensor)
                predicted_index = int(torch.argmax(logits, dim=1).item())
                predicted_label = index_to_class[predicted_index]

            cam = compute_gradcam(
                model=model,
                image_tensor=image_tensor,
                target_layer=target_layer,
                class_index=predicted_index,
            )

            # Redimensiona mapa para o mesmo tamanho da imagem final.
            cam_image = Image.fromarray((cam * 255.0).astype(np.uint8))
            cam_resized = cam_image.resize((args.image_size, args.image_size), resample=Image.BILINEAR)
            cam_resized_np = np.array(cam_resized, dtype=np.float32) / 255.0

            overlay_np = build_overlay_image(original_np, cam_resized_np, alpha=0.45)

            # Salva imagem com nome descritivo para facilitar análise manual.
            safe_label = class_name.replace(' ', '_')
            safe_pred = predicted_label.replace(' ', '_')
            filename = f'{safe_label}_sample{row_idx:02d}_pred_{safe_pred}.png'
            output_path = output_dir / filename
            Image.fromarray(overlay_np).save(output_path)

            metadata_rows.append(
                {
                    'image_path': str(image_path),
                    'true_label': class_name,
                    'predicted_label': predicted_label,
                    'output_file': str(output_path),
                },
            )

    metadata_df = pd.DataFrame(metadata_rows)
    metadata_csv = output_dir / 'gradcam_metadata.csv'
    metadata_df.to_csv(metadata_csv, index=False)

    class_mapping_path = output_dir / 'class_mapping.json'
    with class_mapping_path.open('w', encoding='utf-8') as mapping_file:
        json.dump(class_to_index, mapping_file, indent=2, ensure_ascii=False)

    print('\nGrad-CAM concluído:')
    print(f'- Pasta de imagens: {output_dir.resolve()}')
    print(f'- Metadados: {metadata_csv.resolve()}')
    print(f'- Mapeamento de classes: {class_mapping_path.resolve()}')
    print(f'- Total de visualizações geradas: {len(metadata_df)}')


if __name__ == '__main__':
    # Mantém execução direta do script via linha de comando.
    main()
