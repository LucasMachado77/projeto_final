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

- `src/`: scripts OASIS (`step_01` a `step_08`, mais variantes `06b`, `07b`, `08b` para comparacao sem sobrescrever saidas);
- `src_adni/`: **mesmo pipeline** de passos para o coorte **ADNI** (DICOM + tabelas `study_files` no passo `step_00`; saidas em `reports_adni/`);
- `reports/`: resultados da linha OASIS (`src/`);
- `reports_adni/`: resultados da linha ADNI (`src_adni/`);
- `docs/`: documentos de apoio metodologico;
- `config_adni.local.example.json`: modelo de caminhos ADNI por maquina (copiar para `config_adni.local.json`, nao versionado);
- `requirements.txt`: dependencias do projeto (inclui `pydicom` para o passo ADNI 00).

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

## 7) Pipeline OASIS - ordem recomendada (`src/` e `reports/`)

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

### Etapa 07b - ViT (variante B, pasta e epocas proprias)

Mantém a **Etapa 07** como baseline reprodutivel; use **07b** para novos experimentos (ex.: mais epocas) **sem sobrescrever** `reports/vit_transfer_learning_grouped_binary`.

```bash
python src/step_07b_vit_transfer_learning.py --split-csv reports/split_assignments_grouped_binary.csv --output-dir reports/vit_transfer_learning_grouped_binary_b
```

### Etapa 05 - Comparacao final (baseline vs CNN)

```bash
python src/step_05_compare_results.py --baseline-metrics reports/baseline_ml_grouped_binary/metrics.json --cnn-metrics reports/cnn_transfer_learning_e5_grouped_binary/metrics.json --output-csv reports/final_comparison/comparison_table.csv --output-md reports/final_comparison/comparison_table.md
```

Opcional: incluir **ViT** na mesma tabela.

```bash
python src/step_05_compare_results.py --baseline-metrics reports/baseline_ml_grouped_binary/metrics.json --cnn-metrics reports/cnn_transfer_learning_e5_grouped_binary/metrics.json --vit-metrics reports/vit_transfer_learning_grouped_binary/metrics.json --output-csv reports/final_comparison/comparison_table.csv --output-md reports/final_comparison/comparison_table.md
```

### Etapa 06 - Explicabilidade com Grad-CAM

```bash
python src/step_06_gradcam.py --split-csv reports/split_assignments_grouped_binary.csv --model-path reports/cnn_transfer_learning_e5_grouped_binary/best_model_resnet18.pth --samples-per-class 2 --output-dir reports/gradcam_grouped_binary
```

### Etapa 06b - Grad-CAM estendido (heatmap + overlay, acertos/erros)

Mantém a **Etapa 06** no formato simples (uma PNG por caso); use **06b** para exportar **`_overlay.png`** e **`_heatmap.png`**, selecao **errors_and_correct**, etc., em pasta padrao separada (`reports/gradcam_grouped_binary_b`).

```bash
python src/step_06b_gradcam.py --selection-strategy errors_and_correct --errors-per-class 2 --correct-per-class 2 --output-dir reports/gradcam_clinical_contrast
```

### Etapa 08 - Curvas de aprendizado (generalizacao)

```bash
python src/step_08_plot_learning_curves.py --history-csv reports/cnn_transfer_learning_e5_grouped_binary/history.csv
```

### Etapa 08b - Curvas do ViT (default no history do passo 07)

Mantém a **Etapa 08** com default na CNN; **08b** aponta por padrao para `reports/vit_transfer_learning_grouped_binary/history.csv`.

```bash
python src/step_08b_plot_learning_curves.py
```

## 8) Pipeline ADNI - ordem recomendada (`src_adni/` e `reports_adni/`)

Fluxo paralelo ao OASIS: os passos `step_01` em diante sao os mesmos (baseline, CNN, ViT, Grad-CAM), mas o **passo 00** gera um ImageFolder a partir de **DICOM** (dados em disco apos descompactar o download LONI) e dos CSVs em **`ADNI/study_files/`** (Key MRI + DXSUM). Todas as saidas padrao vao para `reports_adni/` para nao misturar com OASIS.

**Dados esperados (fora deste repositorio):**

- Imagens: por exemplo `projeto_final/ADNI/Data/` (arvore com pastas `ADNI/<PTID>/.../I<image_id>/` e ficheiros `.dcm`);
- Tabelas: `projeto_final/ADNI/study_files/` (ex.: `Clinical_T1w_Imaging_Cohort_Key_MRI_*.csv`, `Clinical_T1w_Imaging_Cohort_DXSUM_*.csv`).

**Caminhos por maquina (sem editar codigo a cada PC):**

1. Copie `config_adni.local.example.json` para **`config_adni.local.json`** na raiz do repo (pasta que contem `src_adni`). Para dados só num disco externo, use **`adni_data_root`** (ex.: `E:/data_lc`); ou preencha `study_files_dir`, `dicom_root`, `output_dir` separadamente. Este ficheiro esta no `.gitignore`.
2. Ou defina **`ADNI_DATA_ROOT`** (ex.: `E:\data_lc`) para o script procurar CSV na raiz / `study_files` e DICOM em `data_lc`, `data_lc\Data` ou `data_lc\ADNI\Data`.
3. Ou variaveis pontuais: `ADNI_DICOM_ROOT`, `ADNI_OUTPUT_DIR`, `ADNI_KEY_MRI_CSV`, `ADNI_DXSUM_CSV`.
4. **Precedencia:** argumentos CLI > env > `config_adni.local.json` > deteccao automatica.

**Layout tipico em `E:\data_lc`:** ficheiros `Clinical_T1w_Imaging_Cohort_*.csv` na raiz (ou em `study_files\`) e a arvore descompactada (ex. pasta `ADNI\`) dentro de `E:\data_lc`. Se o LONI entregou **dois ZIPs** (`Clinical_T1w_Imaging_Cohort_MRI_1` e `_MRI_2`), o `step_00` **indexa ambos** automaticamente quando estão sob `adni_data_root` ou ao lado da pasta `--dicom-root`; também pode usar `dicom_roots` no JSON ou `--extra-dicom-root` para a segunda pasta.

**Etapa 00 - De DICOM + CSVs para pastas `Non Demented` / `Demented` (PNG)**

Cruza Key MRI com DXSUM: alias de visita (`sc` / `scmri` -> `bl` no DXSUM) e, por defeito, **fallback por data** (vizinho clinico mais proximo quando o codigo de visita da imagem nao existe no DXSUM, ex. `v02` / `v04`, ate `--nearest-days`, padrao 120). Por defeito exclui series cuja descricao contem `REPEAT` (use `--keep-repeat-series` para manter).

```bash
cd mei_imagens_medicas_2025
# Com config_adni.local.json preenchido (recomendado em varias maquinas):
python src_adni/step_00_adni_prepare_imagefolder.py
# Ou explicitando pastas:
python src_adni/step_00_adni_prepare_imagefolder.py --dicom-root "C:\Projetos\projeto_final\ADNI\Data" --output-dir "C:\Projetos\projeto_final\ADNI\processed_imagefolder"
```

Opcoes uteis: `--scheme cn_mci_vs_ad` (CN+MCI vs demencia) ou `cn_vs_ad`; `--no-nearest-date`; `--nearest-days 90`; `--max-cases 50` (teste); **`--skip-missing-dicom`** (só gera PNG para `image_id` que existem no disco — recomendado se o CSV tem coorte completa mas só parte dos volumes foi baixada).  
Saidas: `reports_adni/adni_prepare_manifest.csv`, `reports_adni/adni_merge_stats.txt`.

**Etapas 01 a 08 (mesma logica que o OASIS)**

Use `--dataset-dir` apontando para `ADNI/processed_imagefolder` e ficheiros em `reports_adni/`:

```bash
python src_adni/step_01_dataset_check.py --dataset-dir "C:\Projetos\projeto_final\ADNI\processed_imagefolder"
python src_adni/step_02b_group_split_and_report.py --dataset-dir "C:\Projetos\projeto_final\ADNI\processed_imagefolder" --output-csv reports_adni/split_assignments_grouped.csv
python src_adni/step_02c_make_binary_labels.py --input-csv reports_adni/split_assignments_grouped.csv --output-csv reports_adni/split_assignments_grouped_binary.csv
python src_adni/step_03_baseline_ml.py --split-csv reports_adni/split_assignments_grouped_binary.csv --image-size 32 --output-dir reports_adni/baseline_ml_grouped_binary
python src_adni/step_04_cnn_transfer_learning.py --split-csv reports_adni/split_assignments_grouped_binary.csv --epochs 5 --batch-size 32 --output-dir reports_adni/cnn_transfer_learning_e5_grouped_binary
python src_adni/step_07_vit_transfer_learning.py --split-csv reports_adni/split_assignments_grouped_binary.csv --epochs 5 --batch-size 16 --output-dir reports_adni/vit_transfer_learning_grouped_binary
python src_adni/step_05_compare_results.py --baseline-metrics reports_adni/baseline_ml_grouped_binary/metrics.json --cnn-metrics reports_adni/cnn_transfer_learning_e5_grouped_binary/metrics.json --output-csv reports_adni/final_comparison/comparison_table.csv --output-md reports_adni/final_comparison/comparison_table.md
python src_adni/step_06_gradcam.py --split-csv reports_adni/split_assignments_grouped_binary.csv --model-path reports_adni/cnn_transfer_learning_e5_grouped_binary/best_model_resnet18.pth --samples-per-class 2 --output-dir reports_adni/gradcam_grouped_binary
python src_adni/step_08_plot_learning_curves.py --history-csv reports_adni/cnn_transfer_learning_e5_grouped_binary/history.csv
```

Variantes **06b**, **07b**, **08b** existem em `src_adni/` com os mesmos padroes de uso que em `src/`, apontando para `reports_adni/`.

**Notas metodologicas (ADNI):**

- O campo **DIAGNOSIS** do DXSUM segue o esquema habitual do estudo (1=CN, 2=MCI, 3=demencia); documente no relatorio o `--scheme` escolhido.
- O fallback temporal e uma aproximacao; verifique `adni_merge_stats.txt` e o manifest.
- O modelo de treino continua 2D (fatia central); nao e o mesmo workflow que pipelines 3D/registo publicados na literatura ADNI.

## 9) Outputs

### Minimo para reproducao e correcao (OASIS)

- `src/`
- `requirements.txt`
- `README.md`
- `reports/final_comparison/comparison_table.md`
- `reports/final_comparison/comparison_table.csv`
- `reports/*/metrics.json`
- `reports/*/classification_report_test.json`
- `reports/*/confusion_matrix.csv`

### Minimo para reproducao (ADNI)

- `src_adni/`
- `reports_adni/final_comparison/comparison_table.md` (e `.csv`, se gerados)
- `reports_adni/*/metrics.json` e artefactos equivalentes aos do OASIS
- Opcional: `reports_adni/adni_prepare_manifest.csv`, `reports_adni/adni_merge_stats.txt`

### Arquivos auxiliares importantes

- `reports/split_assignments.csv`
- `reports/split_assignments_grouped.csv`
- `reports/split_assignments_grouped_binary.csv`
- `reports_adni/split_assignments_grouped.csv`
- `reports_adni/split_assignments_grouped_binary.csv`
- `docs/roadmap_mestrado_sota.md`

### Nao versionar no envio enxuto

- dados brutos (`data/`, imagens, zips);
- pesos de modelo (`*.pth`, `*.pt`);
- historicos grandes de treino (`history.csv`);
- imagens em massa do Grad-CAM (`*.png`);
- ambiente local (`.venv/`, `__pycache__/`).

## 10) Checklist rapido de reproducao

**OASIS:**

1. Criar e ativar `venv`;
2. instalar dependencias (`pip install -r requirements.txt`);
3. rodar `step_01`, `step_02b`, `step_02c`;
4. treinar `step_03` (baseline), `step_04` (CNN) e `step_07` (ViT);
5. gerar tabela final com `step_05`;
6. validar outputs na pasta `reports/`.

**ADNI:**

1. Colocar DICOM descompactados e CSVs em `projeto_final/ADNI/` (Data + study_files);
2. `pip install -r requirements.txt` (inclui `pydicom`);
3. `python src_adni/step_00_adni_prepare_imagefolder.py` com `--dicom-root` e `--output-dir` corretos;
4. seguir a **secao 8** deste README (`step_01` em `src_adni` ate conclusao, saidas em `reports_adni/`).

## 11) Riscos e boas praticas metodologicas

- priorizar split por paciente para reduzir leakage;
- reportar metricas por classe (nao apenas acuracia global);
- manter reproducibilidade com comando e parametros explicitos;
- documentar claramente diferencas entre cenarios de treino.

## 12) Proximos passos


- migracao de pipeline para `MONAI`;

