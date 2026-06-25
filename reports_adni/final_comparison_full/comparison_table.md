| model | val_accuracy | test_accuracy | test_loss | notes |
| --- | --- | --- | --- | --- |
| Baseline ML | 0.7773 | 0.7888 | - | LogisticRegression com features simples (grayscale + resize) |
| CNN Transfer Learning | 0.7983 | 0.8190 | 0.4819 | ResNet18 com fine-tuning em split por paciente |
| ViT Transfer Learning | 0.8992 | 0.8922 | 0.6270 | ViT-B/16 com transfer learning em split por paciente |
