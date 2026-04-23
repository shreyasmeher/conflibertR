# conflibertR 0.4.0

- Added `conflibert_load()` to load a saved fine-tuned classifier from disk, and a `predict()` method that runs batched inference returning a tidy tibble.
- Added LoRA (parameter-efficient) fine-tuning. Pass `use_lora = TRUE` to `conflibert_finetune()`, `conflibert_compare()`, or `conflibert_active_start()` to train with a low-rank adapter; the adapter is merged into the base model before saving so reloads are transparent. `peft` added to the Python install list.
- Added diversity-aware active learning. Pass `diverse = TRUE` to `conflibert_active_start()` to cluster top-scoring candidates in embedding space and pick one per cluster, preventing near-duplicates from dominating a batch.

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
