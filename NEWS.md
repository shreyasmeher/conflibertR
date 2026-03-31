# conflibertR 0.2.0

- Added `conflibert_finetune()` for training custom classifiers on user data.
- Added `conflibert_compare()` for comparing multiple base models side by side.
- Added `conflibert_benchmark()` for evaluating the pretrained classifier against labeled data.
- Added `conflibert_models()` to list available base model architectures.
- Seven base models supported: ConfliBERT, BERT (uncased/cased), RoBERTa, ModernBERT, DeBERTa v3, DistilBERT.

# conflibertR 0.1.0

- Initial release.
- `conflibert_ner()`: Named Entity Recognition.
- `conflibert_classify()`: Binary classification (conflict vs non-conflict).
- `conflibert_multilabel()`: Multilabel event type classification.
- `conflibert_qa()`: Extractive question answering.
- `conflibert_install()`: One-time Python dependency setup.
