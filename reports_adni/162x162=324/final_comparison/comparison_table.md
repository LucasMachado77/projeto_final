| model | val_accuracy | test_accuracy | test_loss | notes |
| --- | --- | --- | --- | --- |
| Baseline ML | 0.5227 | 0.4545 | - | LogisticRegression com features simples (grayscale + resize) |
| CNN Transfer Learning | 0.6364 | 0.5818 | 0.8709 | ResNet18 com fine-tuning em split por paciente |
