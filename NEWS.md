# conflibertR 0.5.0

## Easier setup

- The backend is now PyTorch-only. TensorFlow and tf-keras are no longer
  installed by default, which makes `conflibert_install()` much smaller
  and more reliable, and removes the `transformers < 4.50` version pin.
- The QA model runs on PyTorch too. Because the published checkpoint only
  ships TensorFlow weights, the first `conflibert_qa()` call converts
  them once and caches the PyTorch copy; environments that already have
  TensorFlow (any 0.4.0 install) do this transparently. Fresh installs
  can opt in with `conflibert_install(qa = TRUE)`, and TensorFlow is
  never used after the one-time conversion.
- New `conflibert_status()`: a one-call diagnostic that checks the Python
  environment, required packages, and compute device, with specific advice
  when something is missing.
- Friendlier errors when the Python backend cannot load, and a startup
  hint when no `conflibert` environment is found.

## Nicer output

- All inference results now print with a themed, readable layout
  (colored labels, probability bars, entity highlighting in the source
  sentence). They are still plain tibbles underneath, so all existing
  code (`$`, `dplyr`, `[`, etc.) keeps working unchanged.
- `conflibert_finetune()` results have a class with a clean `print()`
  method; all fields are accessed exactly as before.
- `print()` for active-learning sessions and loaded classifiers got the
  same treatment.

## New plots

- New `theme_conflibert()` ggplot2 theme: a modern, flat look (no tick
  marks, hairline grids along the data axis only, bold left-aligned
  titles) used by all package plots and exported for your own figures.
  A `grid` argument controls which grid lines are kept.
- Every result object now plots itself, and the default views are
  aggregated, built for real corpus sizes:
  - classification results: texts per class with average confidence
  - NER results: entity counts by type, or the most frequent entities
    (`type = "entities"`)
  - multilabel results: share of texts flagged per category (a single
    text gets its category probabilities instead)
  - model comparisons: ranked metric dot chart
  - fine-tune results: test-set confusion matrix
  - active-learning sessions: learning curve plus uncertainty trend
- Each is available both as base-graphics `plot()` (no extra packages)
  and as `ggplot2::autoplot()` (returns a customizable ggplot).

## Faster and more capable inference

- `conflibert_classify()`, `conflibert_multilabel()`, and
  `conflibert_ner()` now run batched inference (one Python call per batch
  instead of per text), with a progress bar for large inputs and a
  heads-up message the first time a model is downloaded.
- `conflibert_ner()` gains `score`, `start`, and `end` columns (1-based
  character offsets into the input text).
- `conflibert_qa()` is vectorized (with recycling) and gains a
  `details = TRUE` argument returning answer, confidence score, and
  character span. The single-question default still returns a plain
  string, exactly as before.
- `conflibert_compare()` no longer fails when one model errors; the
  failed model gets an `error` column and the rest are kept.

## Internal

- Added a testthat suite (63 tests) that runs against a mocked Python
  backend, so R-side logic is tested without torch or model downloads.

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
