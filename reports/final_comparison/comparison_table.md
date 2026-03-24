| model | val_accuracy | test_accuracy | test_loss | notes |
| --- | --- | --- | --- | --- |
| Baseline ML | 0.5997 | 0.7533 | - | LogisticRegression com features simples (grayscale + resize) |
| CNN Transfer Learning | 0.8249 | 0.8048 | 1.7486 | ResNet18 com fine-tuning em split por paciente |
