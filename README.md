# Projeto 2025/2026 MEI - Classificacao de Imagens Medicas

Projeto academico para comparacao de tecnicas de classificacao em neuroimagem com foco em demencia, cobrindo baseline classico, CNN, ViT e explicabilidade com Grad-CAM.

## 1) Objetivo do projeto

Avaliar e comparar abordagens de:
- `Machine Learning` classico (`LogisticRegression`);
- `Deep Learning` com `ResNet18` (transfer learning);
- arquitetura moderna `Vision Transformer` (`ViT`);
- interpretabilidade via `xAI` com `Grad-CAM`.

O fluxo principal usa **split por paciente** (`subject_id`) e cenario **binario** (`Demented` vs `Non Demented`) para reduzir leakage e aproximar melhor a generalizacao.

## 2) Resultados principais

| Modelo | Val Accuracy | Test Accuracy | Test Loss | Observacao |
| --- | --- | --- | --- | --- |
| Baseline ML | 0.5997 | 0.7533 | - | `LogisticRegression` com features simples |
| CNN Transfer Learning (ResNet18) | 0.8249 | 0.8048 | 1.7486 | Melhor desempenho no teste |
| ViT Transfer Learning | 0.6628 | 0.7282 | 0.5206 | Abaixo da CNN nesta configuracao |

Arquivos oficiais de consolidacao:
- `reports/final_comparison/comparison_table.md`
- `reports/final_comparison/comparison_table.csv`

## 3) Estrutura do repositorio

- `src/`: scripts de execucao por etapa (`step_01` a `step_07`);
- `reports/`: resultados, metricas, matrizes e comparacoes;
- `docs/`: documentos de apoio metodologico;
- `requirements.txt`: dependencias do projeto.

## 4) Requisitos

- Python 3.10+ (recomendado);
- `pip` atualizado;
- sistema com memoria suficiente para treino (CPU funciona, GPU acelera).

## 5) Instalacao do ambiente

```bash
python -m venv .venv
```

```bash
# Windows PowerShell
.venv\Scripts\Activate.ps1
```

```bash
pip install -r requirements.txt
```

## 6) Dataset e formato esperado

O dataset **nao** e versionado no repositorio.  
Informe o caminho local com `--dataset-dir` nos scripts.

Fonte oficial do OASIS-1:
- [OASIS Brains - OASIS-1](https://sites.wustl.edu/oasisbrains/home/oasis-1/)

Arquivos de imagem OASIS-1 (cada download ~1.5 GB):
- `oasis_cross-sectional_disc1.tar.gz`
- `oasis_cross-sectional_disc2.tar.gz`
- `oasis_cross-sectional_disc3.tar.gz`
- `oasis_cross-sectional_disc4.tar.gz`
- `oasis_cross-sectional_disc5.tar.gz`
- `oasis_cross-sectional_disc6.tar.gz`
- `oasis_cross-sectional_disc7.tar.gz`
- `oasis_cross-sectional_disc8.tar.gz`
- `oasis_cross-sectional_disc9.tar.gz`
- `oasis_cross-sectional_disc10.tar.gz`
- `oasis_cross-sectional_disc11.tar.gz`
- `oasis_cross-sectional_disc12.tar.gz`

Observacao: baixe todos os discos, extraia em uma pasta local e use esse caminho no parametro `--dataset-dir`.

Exemplo de chamada:

```bash
python src/step_01_dataset_check.py --dataset-dir "C:\caminho\para\dataset"
```

## 7) Pipeline completo (ordem recomendada)

### Etapa 01 - Validacao inicial do dataset

```bash
python src/step_01_dataset_check.py --dataset-dir "C:\caminho\para\dataset"
```

Saida esperada: relatorio de classes e consistencia estrutural.

### Etapa 02 - Split estratificado padrao

```bash
python src/step_02_split_and_report.py --dataset-dir "C:\caminho\para\dataset" --output-csv reports/split_assignments.csv
```

### Etapa 02b - Split por paciente (recomendado)

```bash
python src/step_02b_group_split_and_report.py --dataset-dir "C:\caminho\para\dataset" --output-csv reports/split_assignments_grouped.csv
```

### Etapa 02c - Conversao para labels binarias

```bash
python src/step_02c_make_binary_labels.py --input-csv reports/split_assignments_grouped.csv --output-csv reports/split_assignments_grouped_binary.csv
```

### Etapa 03 - Baseline classico (ML)

```bash
python src/step_03_baseline_ml.py --split-csv reports/split_assignments_grouped_binary.csv --image-size 32 --output-dir reports/baseline_ml_grouped_binary
```

### Etapa 04 - CNN (ResNet18 transfer learning)

```bash
python src/step_04_cnn_transfer_learning.py --split-csv reports/split_assignments_grouped_binary.csv --epochs 5 --batch-size 32 --output-dir reports/cnn_transfer_learning_e5_grouped_binary
```

### Etapa 07 - ViT (transfer learning)

```bash
python src/step_07_vit_transfer_learning.py --split-csv reports/split_assignments_grouped_binary.csv --epochs 5 --batch-size 16 --output-dir reports/vit_transfer_learning_grouped_binary
```

### Etapa 05 - Comparacao final (baseline vs CNN)

```bash
python src/step_05_compare_results.py --baseline-metrics reports/baseline_ml_grouped_binary/metrics.json --cnn-metrics reports/cnn_transfer_learning_e5_grouped_binary/metrics.json --output-csv reports/final_comparison/comparison_table.csv --output-md reports/final_comparison/comparison_table.md
```

### Etapa 06 - Explicabilidade com Grad-CAM

```bash
python src/step_06_gradcam.py --split-csv reports/split_assignments_grouped_binary.csv --model-path reports/cnn_transfer_learning_e5_grouped_binary/best_model_resnet18.pth --samples-per-class 2 --output-dir reports/gradcam_grouped_binary
```

## 8) Outputs

### Minimo para reproducao e correcao

- `src/`
- `requirements.txt`
- `README.md`
- `reports/final_comparison/comparison_table.md`
- `reports/final_comparison/comparison_table.csv`
- `reports/*/metrics.json`
- `reports/*/classification_report_test.json`
- `reports/*/confusion_matrix.csv`

### Arquivos auxiliares importantes

- `reports/split_assignments.csv`
- `reports/split_assignments_grouped.csv`
- `reports/split_assignments_grouped_binary.csv`
- `docs/roadmap_mestrado_sota.md`

### Nao versionar no envio enxuto

- dados brutos (`data/`, imagens, zips);
- pesos de modelo (`*.pth`, `*.pt`);
- historicos grandes de treino (`history.csv`);
- imagens em massa do Grad-CAM (`*.png`);
- ambiente local (`.venv/`, `__pycache__/`).

## 9) Checklist rapido de reproducao

1. Criar e ativar `venv`;
2. instalar dependencias (`pip install -r requirements.txt`);
3. rodar `step_01`, `step_02b`, `step_02c`;
4. treinar `step_03` (baseline), `step_04` (CNN) e `step_07` (ViT);
5. gerar tabela final com `step_05`;
6. validar outputs na pasta `reports/`.

## 10) Riscos e boas praticas metodologicas

- priorizar split por paciente para reduzir leakage;
- reportar metricas por classe (nao apenas acuracia global);
- manter reproducibilidade com comando e parametros explicitos;
- documentar claramente diferencas entre cenarios de treino.

## 11) Proximos passos

Consulte `docs/roadmap_mestrado_sota.md` para:
- migracao de pipeline para `MONAI`;
- 
