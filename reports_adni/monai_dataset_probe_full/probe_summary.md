# ADNI MONAI dataset probe

- source: `split_csv`
- sampled_records: `12`
- MONAI: `1.6.0`
- torch: `2.12.1+cpu`
- batch image shape: `[4, 1, 224, 224]`
- intensity range: `0.0` to `1.0`

## Class mapping

- `Demented` -> `0`
- `Non Demented` -> `1`

## Path check

- selected split rows: `1024`
- existing image paths: `1024`
- missing image paths: `0`
