from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    # Permite apontar qualquer history.csv gerado pelos scripts de treino.
    parser = argparse.ArgumentParser(
        description='Plota curvas de treino/validação e estima época de perda de generalização.',
    )
    parser.add_argument(
        '--history-csv',
        type=Path,
        default=Path('reports_adni/cnn_transfer_learning_e5_grouped_binary/history.csv'),
        help='CSV com colunas epoch, train_loss, val_loss, train_accuracy, val_accuracy.',
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=None,
        help='Pasta de saída (padrão: mesmo diretório do history.csv).',
    )
    parser.add_argument(
        '--output-basename',
        type=str,
        default='learning_curves',
        help='Nome base dos arquivos PNG/JSON (sem extensão).',
    )
    return parser.parse_args()


def load_history(history_csv: Path) -> pd.DataFrame:
    # Valida colunas esperadas para evitar gráficos silenciosamente incorretos.
    if not history_csv.exists():
        raise FileNotFoundError(f'History não encontrado: {history_csv}')
    history_df = pd.read_csv(history_csv)
    required = {'epoch', 'train_loss', 'val_loss', 'train_accuracy', 'val_accuracy'}
    missing = required.difference(history_df.columns)
    if missing:
        raise ValueError(f'Colunas ausentes no history: {sorted(missing)}')
    return history_df.sort_values('epoch').reset_index(drop=True)


def find_first_overfit_epoch(history_df: pd.DataFrame) -> int | None:
    # Heurística: primeira época e>1 em que val_loss sobe vs e-1 e train_loss desce vs e-1.
    # Indica divergência típica “treino melhora / validação piora”.
    if len(history_df) < 2:
        return None
    for index in range(1, len(history_df)):
        prev = history_df.iloc[index - 1]
        cur = history_df.iloc[index]
        if float(cur['val_loss']) > float(prev['val_loss']) and float(cur['train_loss']) < float(prev['train_loss']):
            return int(cur['epoch'])
    return None


def find_best_val_epochs(history_df: pd.DataFrame) -> dict[str, float | int]:
    # Registra épocas ótimas por loss e por acurácia de validação.
    best_loss_row = history_df.loc[history_df['val_loss'].astype(float).idxmin()]
    best_acc_row = history_df.loc[history_df['val_accuracy'].astype(float).idxmax()]
    return {
        'best_val_loss_epoch': int(best_loss_row['epoch']),
        'best_val_loss': float(best_loss_row['val_loss']),
        'best_val_accuracy_epoch': int(best_acc_row['epoch']),
        'best_val_accuracy': float(best_acc_row['val_accuracy']),
    }


def build_summary(history_df: pd.DataFrame) -> dict[str, object]:
    # Consolida heurísticas em um JSON útil para o relatório.
    summary: dict[str, object] = {
        'rows': len(history_df),
        **find_best_val_epochs(history_df),
        'first_overfit_epoch_heuristic': find_first_overfit_epoch(history_df),
    }
    return summary


def plot_curves(history_df: pd.DataFrame, output_png: Path, title: str) -> None:
    # Duas subfiguras: loss e acurácia, treino vs validação.
    output_png.parent.mkdir(parents=True, exist_ok=True)
    epochs = history_df['epoch'].astype(float).tolist()

    figure, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(epochs, history_df['train_loss'], label='train_loss', marker='o')
    axes[0].plot(epochs, history_df['val_loss'], label='val_loss', marker='o')
    axes[0].set_title('Loss por época')
    axes[0].set_xlabel('Época')
    axes[0].set_ylabel('Loss')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(epochs, history_df['train_accuracy'], label='train_accuracy', marker='o')
    axes[1].plot(epochs, history_df['val_accuracy'], label='val_accuracy', marker='o')
    axes[1].set_title('Acurácia por época')
    axes[1].set_xlabel('Época')
    axes[1].set_ylabel('Acurácia')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    figure.suptitle(title)
    figure.tight_layout()
    figure.savefig(output_png, dpi=150)
    plt.close(figure)


def main() -> None:
    # Gera artefatos visuais + JSON interpretável para discussão de generalização.
    args = parse_args()
    history_df = load_history(args.history_csv)

    output_dir = args.output_dir or args.history_csv.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    output_png = output_dir / f'{args.output_basename}.png'
    output_json = output_dir / f'{args.output_basename}_summary.json'

    title = f'Curvas de aprendizado — {args.history_csv.parent.name}'
    plot_curves(history_df=history_df, output_png=output_png, title=title)
    summary = build_summary(history_df)

    with output_json.open('w', encoding='utf-8') as summary_file:
        json.dump(summary, summary_file, indent=2, ensure_ascii=False)

    print('\nCurvas de aprendizado geradas:')
    print(f'- PNG: {output_png.resolve()}')
    print(f'- Resumo: {output_json.resolve()}')
    print('\nResumo (JSON):')
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    # Execução direta via linha de comando.
    main()
