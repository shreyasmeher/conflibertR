# conflibertR 0.3.1

- Added `conflibert_active_label()`: opens a Shiny gadget for labeling the current query, matching the GUI's point-and-click experience. Requires `shiny` and `miniUI` (Suggests).

# conflibertR 0.3.0

- Added an active-learning workflow for efficient labeling of unlabeled pools.
- `conflibert_active_start()`: train on a labeled seed and return the most uncertain samples from a pool.
- `conflibert_active_next()`: submit labels for the current query, retrain, and get the next batch.
- `conflibert_active_save()`: persist the active-learning model as a HuggingFace checkpoint.
- Session objects have `print()` and `plot()` methods for quick inspection and tracking metrics across rounds.
- Three query strategies supported: `entropy`, `margin`, `least_confidence`.
- Bundled active-learning example dataset available via `conflibert_example("active")`.

# conflibertR 0.2.0

- Added `conflibert_finetune()` for training custom classifiers on user data.
- Added `conflibert_compare()` for comparing multiple base models side by side.
- Added `conflibert_benchmark()` for evaluating the pretrained classifier against labeled data.
- Added `conflibert_models()` to list available base model architectures.
- Seven base models supported: ConfliBERT, BERT (uncased/cased), RoBERTa, ModernBERT, DeBERTa v3, DistilBERT.
- Added `conflibert_example()` to load bundled example datasets (binary and multiclass) for quick testing.

# conflibertR 0.1.0

- Initial release.
- `conflibert_ner()`: Named Entity Recognition.
- `conflibert_classify()`: Binary classification (conflict vs non-conflict).
- `conflibert_multilabel()`: Multilabel event type classification.
- `conflibert_qa()`: Extractive question answering.
- `conflibert_install()`: One-time Python dependency setup.
