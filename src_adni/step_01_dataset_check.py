from __future__ import annotations

import argparse
from pathlib import Path


VALID_IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}


def count_images_in_class(class_dir: Path) -> int:
    # Conta imagens válidas dentro de uma pasta de classe.
    return sum(
        1
        for file_path in class_dir.rglob('*')
        if file_path.is_file() and file_path.suffix.lower() in VALID_IMAGE_EXTENSIONS
    )


def summarize_dataset(dataset_dir: Path) -> list[tuple[str, int]]:
    # Lê subpastas do dataset como classes e retorna a contagem por classe.
    class_dirs = [path for path in dataset_dir.iterdir() if path.is_dir()]
    class_dirs.sort(key=lambda path: path.name.lower())

    summary: list[tuple[str, int]] = []
    for class_dir in class_dirs:
        image_count = count_images_in_class(class_dir)
        summary.append((class_dir.name, image_count))

    return summary


def print_summary(summary: list[tuple[str, int]]) -> None:
    # Exibe resumo amigável para facilitar a inspeção inicial dos dados.
    total_images = sum(image_count for _, image_count in summary)
    print('\nResumo do dataset por classe:\n')

    if not summary:
        print('Nenhuma classe foi encontrada (sem subpastas no diretório informado).')
        return

    for class_name, image_count in summary:
        print(f'- {class_name}: {image_count} imagens')

    print(f'\nTotal de imagens: {total_images}')


def parse_args() -> argparse.Namespace:
    # Recebe o caminho do dataset para permitir reuso do script em diferentes fontes.
    parser = argparse.ArgumentParser(
        description='Valida e resume a estrutura de um dataset de imagens médicas.',
    )
    parser.add_argument(
        '--dataset-dir',
        type=Path,
        required=True,
        help='Caminho para a pasta principal do dataset.',
    )
    return parser.parse_args()


def main() -> None:
    # Executa validações básicas antes da leitura para evitar erros silenciosos.
    args = parse_args()
    dataset_dir: Path = args.dataset_dir

    if not dataset_dir.exists():
        raise FileNotFoundError(f'O diretório não existe: {dataset_dir}')

    if not dataset_dir.is_dir():
        raise NotADirectoryError(f'O caminho informado não é uma pasta: {dataset_dir}')

    summary = summarize_dataset(dataset_dir)
    print_summary(summary)


if __name__ == '__main__':
    # Mantém o script executável diretamente pela linha de comando.
    main()
