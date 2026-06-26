"""ConfliBERT inference module for the R package.

Models are loaded lazily and cached so the first call to each task
downloads and loads the model, and subsequent calls reuse it.
"""

import os

import torch
import numpy as np
import gc
import tempfile

_models = {}

_BATCH_SIZE = 32

MODEL_MAP = {
    "ConfliBERT": "snowood1/ConfliBERT-scr-uncased",
    "BERT Base Uncased": "bert-base-uncased",
    "BERT Base Cased": "bert-base-cased",
    "RoBERTa Base": "roberta-base",
    "ModernBERT Base": "answerdotai/ModernBERT-base",
    "DeBERTa v3 Base": "microsoft/deberta-v3-base",
    "DistilBERT Base": "distilbert-base-uncased",
}

AVAILABLE_MODELS = list(MODEL_MAP.keys())


def _get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    # MPS (Apple Silicon) can have compatibility issues with some models,
    # and CPU is fast enough for BERT-class inference. Use CUDA when
    # available, otherwise CPU.
    return torch.device("cpu")


def _load(task):
    if task in _models:
        return _models[task]

    from transformers import AutoTokenizer

    device = _get_device()

    if task == "ner":
        from transformers import AutoModelForTokenClassification
        name = "eventdata-utd/conflibert-named-entity-recognition"
        model = AutoModelForTokenClassification.from_pretrained(name).to(device)
        tokenizer = AutoTokenizer.from_pretrained(name)
    elif task == "classify":
        from transformers import AutoModelForSequenceClassification
        name = "eventdata-utd/conflibert-binary-classification"
        model = AutoModelForSequenceClassification.from_pretrained(name).to(device)
        tokenizer = AutoTokenizer.from_pretrained(name)
    elif task == "multilabel":
        from transformers import AutoModelForSequenceClassification
        name = "eventdata-utd/conflibert-satp-relevant-multilabel"
        model = AutoModelForSequenceClassification.from_pretrained(name).to(device)
        tokenizer = AutoTokenizer.from_pretrained(name)
    elif task == "qa":
        model, tokenizer = _load_qa(device)
    else:
        raise ValueError(f"Unknown task: {task}")

    _models[task] = (model, tokenizer)
    return model, tokenizer


def is_loaded(task):
    """True if the model for `task` is already loaded in this session."""
    return task in _models


def _qa_cache_dir():
    base = os.environ.get("XDG_CACHE_HOME") or os.path.join(
        os.path.expanduser("~"), ".cache"
    )
    return os.path.join(base, "conflibertR", "ConfliBERT-QA-pt")


def _load_qa(device):
    """Load the QA model in PyTorch.

    The published checkpoint only ships TensorFlow weights, so the first
    load converts them to PyTorch (requires TensorFlow once) and caches
    the converted copy; every later load is pure PyTorch.
    """
    from transformers import AutoTokenizer, AutoModelForQuestionAnswering

    name = "salsarra/ConfliBERT-QA"
    cache = _qa_cache_dir()

    has_cached_weights = any(
        os.path.exists(os.path.join(cache, f))
        for f in ("model.safetensors", "pytorch_model.bin")
    )
    if has_cached_weights:
        model = AutoModelForQuestionAnswering.from_pretrained(cache).to(device)
        tokenizer = AutoTokenizer.from_pretrained(cache)
        return model, tokenizer

    try:
        model = AutoModelForQuestionAnswering.from_pretrained(name).to(device)
        tokenizer = AutoTokenizer.from_pretrained(name)
        return model, tokenizer
    except (OSError, EnvironmentError):
        pass

    try:
        import tensorflow  # noqa: F401
    except ImportError as e:
        raise RuntimeError(
            "The ConfliBERT QA checkpoint only publishes TensorFlow "
            "weights, and no converted copy was found locally. Install "
            "TensorFlow once to convert it:\n"
            "  conflibert_install(qa = TRUE)\n"
            "After the first successful QA call the converted PyTorch "
            "weights are cached and TensorFlow is never needed again."
        ) from e

    # convert the TF checkpoint manually: from_pretrained(from_tf=True)
    # initializes on the meta device in recent transformers and yields a
    # weightless model
    from transformers import AutoConfig
    from transformers.modeling_tf_pytorch_utils import (
        load_tf2_checkpoint_in_pytorch_model,
    )
    from huggingface_hub import hf_hub_download

    config = AutoConfig.from_pretrained(name)
    model = AutoModelForQuestionAnswering.from_config(config)
    h5 = hf_hub_download(name, "tf_model.h5")
    model = load_tf2_checkpoint_in_pytorch_model(
        model, h5, allow_missing_keys=True
    )
    tokenizer = AutoTokenizer.from_pretrained(name)
    os.makedirs(cache, exist_ok=True)
    model.save_pretrained(cache)
    tokenizer.save_pretrained(cache)
    return model.to(device), tokenizer


def ner_batch(texts):
    """Run NER over a list of texts.

    Returns a columnar dict with doc_id (1-based), entity, label, score,
    start, end. Offsets are 1-based inclusive character positions into the
    original text (ready for R's substr()).
    """
    model, tokenizer = _load("ner")
    device = _get_device()
    texts = list(texts)

    out = {"doc_id": [], "entity": [], "label": [],
           "score": [], "start": [], "end": []}

    for lo in range(0, len(texts), _BATCH_SIZE):
        batch = texts[lo:lo + _BATCH_SIZE]
        enc = tokenizer(
            batch, return_tensors="pt", truncation=True, padding=True, max_length=512,
            return_offsets_mapping=True, return_special_tokens_mask=True,
        )
        offsets = enc.pop("offset_mapping").tolist()
        special = enc.pop("special_tokens_mask").tolist()
        inputs = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=2).cpu().numpy()
        tag_ids = probs.argmax(axis=2)
        attn = enc["attention_mask"].tolist()

        for b, text in enumerate(batch):
            tokens = tokenizer.convert_ids_to_tokens(enc["input_ids"][b].tolist())
            ent = None  # [label, start, end, [probs]]

            def close(e):
                if e is None:
                    return
                out["doc_id"].append(lo + b + 1)
                out["entity"].append(text[e[1]:e[2]])
                out["label"].append(e[0])
                out["score"].append(round(float(np.mean(e[3])), 4))
                out["start"].append(e[1] + 1)
                out["end"].append(e[2])

            for i, token in enumerate(tokens):
                if not attn[b][i] or special[b][i]:
                    continue
                label = model.config.id2label[int(tag_ids[b][i])].split("-")[-1]
                p = float(probs[b][i][tag_ids[b][i]])
                o_start, o_end = offsets[b][i]

                if token.startswith("##"):
                    # continuation word-piece: stays with the open entity
                    if ent is not None:
                        ent[2] = o_end
                        ent[3].append(p)
                elif label != "O":
                    if ent is not None and label == ent[0]:
                        ent[2] = o_end
                        ent[3].append(p)
                    else:
                        close(ent)
                        ent = [label, o_start, o_end, [p]]
                else:
                    close(ent)
                    ent = None
            close(ent)

    return out


def classify_batch(texts, batch_size=_BATCH_SIZE):
    """Binary classification over a list of texts.

    Returns a columnar dict with label, class, confidence, prob_negative,
    prob_positive (one element per input text).
    """
    model, tokenizer = _load("classify")
    device = _get_device()
    texts = list(texts)

    out = {"label": [], "class": [], "confidence": [],
           "prob_negative": [], "prob_positive": []}

    for lo in range(0, len(texts), int(batch_size)):
        batch = texts[lo:lo + int(batch_size)]
        inputs = tokenizer(
            batch, return_tensors="pt", truncation=True, padding=True, max_length=512,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        preds = probs.argmax(axis=1)

        for i in range(len(batch)):
            p = int(preds[i])
            out["label"].append("Positive" if p == 1 else "Negative")
            out["class"].append(p)
            out["confidence"].append(round(float(probs[i][p]), 4))
            out["prob_negative"].append(round(float(probs[i][0]), 4))
            out["prob_positive"].append(round(float(probs[i][1]), 4))

    return out


MULTILABEL_CATEGORIES = [
    "Armed Assault", "Bombing or Explosion", "Kidnapping", "Other",
]


def multilabel_batch(texts, batch_size=_BATCH_SIZE):
    """Multilabel classification over a list of texts.

    Returns a dict with 'categories' and 'probabilities' (one row of
    category probabilities per input text).
    """
    model, tokenizer = _load("multilabel")
    device = _get_device()
    texts = list(texts)

    rows = []
    for lo in range(0, len(texts), int(batch_size)):
        batch = texts[lo:lo + int(batch_size)]
        inputs = tokenizer(
            batch, return_tensors="pt", truncation=True, padding=True, max_length=512,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = model(**inputs).logits
        probs = torch.sigmoid(logits).cpu().numpy()
        rows.extend([[round(float(x), 4) for x in row] for row in probs])

    return {"categories": MULTILABEL_CATEGORIES, "probabilities": rows}


def qa(context, question):
    """Extractive question answering (PyTorch).

    Returns a dict with answer, score, start, end. Offsets are 1-based
    inclusive character positions into the context.
    """
    model, tokenizer = _load("qa")
    device = _get_device()

    enc = tokenizer(
        question, context, return_tensors="pt", truncation=True,
        max_length=512, return_offsets_mapping=True,
    )
    offsets = enc.pop("offset_mapping")[0].tolist()
    seq_ids = enc.sequence_ids(0)
    inputs = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        out = model(**inputs)

    start_logits = out.start_logits[0].cpu()
    end_logits = out.end_logits[0].cpu()

    # only consider positions inside the context passage
    mask = torch.tensor(
        [0.0 if s == 1 else float("-inf") for s in seq_ids]
    )
    start_logits = start_logits + mask
    end_logits = end_logits + mask

    start = int(torch.argmax(start_logits))
    end_tail = end_logits.clone()
    end_tail[:start] = float("-inf")
    end = int(torch.argmax(end_tail))

    p_start = torch.softmax(start_logits, dim=0)[start]
    p_end = torch.softmax(end_logits, dim=0)[end]
    score = float(p_start * p_end)

    char_start, char_end = offsets[start][0], offsets[end][1]
    return {
        "answer": context[char_start:char_end],
        "score": round(score, 4),
        "start": int(char_start) + 1,
        "end": int(char_end),
    }


# --- single-text wrappers kept for backward compatibility ---------------

def ner(text):
    """Run NER on a single text. Returns a list of dicts."""
    r = ner_batch([text])
    return [
        {"entity": r["entity"][i], "label": r["label"][i],
         "score": r["score"][i], "start": r["start"][i], "end": r["end"][i]}
        for i in range(len(r["entity"]))
    ]


def classify(text):
    """Binary classification of a single text. Returns a dict."""
    r = classify_batch([text])
    return {k: v[0] for k, v in r.items()}


def multilabel(text):
    """Multilabel classification of a single text. Returns a list of dicts."""
    r = multilabel_batch([text])
    probs = r["probabilities"][0]
    return [
        {"label": r["categories"][i], "probability": probs[i],
         "predicted": bool(probs[i] >= 0.5)}
        for i in range(len(probs))
    ]


# =========================================================================
# Benchmarking & Fine-tuning
# =========================================================================

def benchmark(texts, labels):
    """Evaluate the pretrained binary classifier against labeled data.

    Returns a dict with accuracy, precision, recall, f1, n.
    """
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
    )

    texts = list(texts)
    labels = [int(l) for l in labels]
    predictions = classify_batch(texts)["class"]

    return {
        "accuracy": round(accuracy_score(labels, predictions), 4),
        "precision": round(precision_score(labels, predictions, zero_division=0), 4),
        "recall": round(recall_score(labels, predictions, zero_division=0), 4),
        "f1": round(f1_score(labels, predictions, zero_division=0), 4),
        "n": len(labels),
    }


class _Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = [int(l) for l in labels]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def _apply_lora(model, lora_rank, lora_alpha):
    """Wrap a classification head model with a LoRA adapter."""
    try:
        from peft import LoraConfig, TaskType, get_peft_model
    except ImportError as e:
        raise ImportError(
            "LoRA fine-tuning requires the `peft` package. "
            "Reinstall the Python backend with conflibert_install()."
        ) from e
    lora_cfg = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=int(lora_rank),
        lora_alpha=int(lora_alpha),
        lora_dropout=0.1,
        bias="none",
    )
    model.enable_input_require_grads()
    return get_peft_model(model, lora_cfg)


def _merge_lora(model):
    """Merge a LoRA adapter into the base model so saves are vanilla checkpoints."""
    if hasattr(model, "merge_and_unload"):
        return model.merge_and_unload()
    return model


def _make_metrics_fn(task_type):
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
    )

    def compute(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        acc = accuracy_score(labels, preds)
        if task_type == "binary":
            return {
                "accuracy": acc,
                "precision": precision_score(labels, preds, zero_division=0),
                "recall": recall_score(labels, preds, zero_division=0),
                "f1": f1_score(labels, preds, zero_division=0),
            }
        return {
            "accuracy": acc,
            "f1_macro": f1_score(labels, preds, average="macro", zero_division=0),
            "f1_micro": f1_score(labels, preds, average="micro", zero_division=0),
            "precision_macro": precision_score(
                labels, preds, average="macro", zero_division=0
            ),
            "recall_macro": recall_score(
                labels, preds, average="macro", zero_division=0
            ),
        }

    return compute


def finetune(
    train_texts, train_labels, dev_texts, dev_labels,
    test_texts, test_labels, model_name="ConfliBERT",
    task="binary", epochs=3, batch_size=8, lr=2e-5,
    save_dir=None, use_lora=False, lora_rank=8, lora_alpha=16,
    seed=42,
):
    """Fine-tune a classification model and evaluate on the test set.

    `seed` seeds Python, NumPy, and PyTorch so the random classifier-head
    initialization, data shuffling, and dropout are reproducible: two runs
    with the same seed on the same hardware and library versions produce
    the same model and metrics.

    Returns a dict with metrics, runtime, predictions, probabilities,
    true_labels, and model_dir.
    """
    from transformers import (
        AutoTokenizer,
        AutoModelForSequenceClassification,
        TrainingArguments,
        Trainer,
        EarlyStoppingCallback,
        set_seed,
    )

    # seed before the model is built so the randomly initialized
    # classification head is reproducible (the Trainer only re-seeds at
    # train() time, after the head already exists)
    set_seed(int(seed))

    model_id = MODEL_MAP.get(model_name, model_name)
    train_texts = list(train_texts)
    dev_texts = list(dev_texts)
    test_texts = list(test_texts)
    train_labels = [int(l) for l in train_labels]
    dev_labels = [int(l) for l in dev_labels]
    test_labels = [int(l) for l in test_labels]
    num_labels = max(max(train_labels), max(dev_labels), max(test_labels)) + 1
    if task == "binary":
        num_labels = 2

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_id, num_labels=num_labels,
    )
    if use_lora:
        model = _apply_lora(model, lora_rank, lora_alpha)

    train_ds = _Dataset(
        tokenizer(train_texts, truncation=True, padding=True, max_length=512),
        train_labels,
    )
    dev_ds = _Dataset(
        tokenizer(dev_texts, truncation=True, padding=True, max_length=512),
        dev_labels,
    )
    test_ds = _Dataset(
        tokenizer(test_texts, truncation=True, padding=True, max_length=512),
        test_labels,
    )

    output_dir = save_dir or tempfile.mkdtemp(prefix="conflibert_r_")
    best_metric = "f1" if task == "binary" else "f1_macro"

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=int(epochs),
        per_device_train_batch_size=int(batch_size),
        per_device_eval_batch_size=int(batch_size) * 2,
        learning_rate=float(lr),
        weight_decay=0.01,
        warmup_ratio=0.1,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model=best_metric,
        greater_is_better=True,
        save_total_limit=1,
        report_to="none",
        seed=int(seed),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        compute_metrics=_make_metrics_fn(task),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    train_result = trainer.train()
    pred_output = trainer.predict(test_ds)
    logits = pred_output.predictions
    preds = np.argmax(logits, axis=-1).tolist()
    probs = torch.softmax(torch.tensor(logits), dim=1).numpy().tolist()

    test_results = trainer.evaluate(test_ds, metric_key_prefix="test")
    metrics = {}
    for k, v in test_results.items():
        if isinstance(v, (int, float)) and "epoch" not in k:
            metrics[k.replace("test_", "")] = round(float(v), 4)

    runtime = round(train_result.metrics.get("train_runtime", 0), 1)

    final_model = trainer.model
    if use_lora:
        final_model = _merge_lora(final_model)

    if save_dir:
        final_model.save_pretrained(save_dir)
        tokenizer.save_pretrained(save_dir)

    return {
        "metrics": metrics,
        "runtime": runtime,
        "predictions": preds,
        "probabilities": probs,
        "true_labels": test_labels,
        "model_dir": output_dir,
    }


def compare(
    train_texts, train_labels, dev_texts, dev_labels,
    test_texts, test_labels, model_names,
    task="binary", epochs=3, batch_size=8, lr=2e-5,
    use_lora=False, lora_rank=8, lora_alpha=16, seed=42,
):
    """Fine-tune multiple models on the same data and compare performance.

    Every model is trained from the same `seed`, so the comparison is
    reproducible run to run.

    Returns a list of dicts, one per model, each with 'model', metrics, and
    'runtime' keys.
    """
    model_names = list(model_names)
    results = []

    for name in model_names:
        try:
            result = finetune(
                train_texts, train_labels,
                dev_texts, dev_labels,
                test_texts, test_labels,
                model_name=name, task=task,
                epochs=epochs, batch_size=batch_size, lr=lr,
                use_lora=use_lora, lora_rank=lora_rank, lora_alpha=lora_alpha,
                seed=seed,
            )
            row = {"model": name, "runtime": result["runtime"]}
            row.update(result["metrics"])
            results.append(row)
        except Exception as e:
            results.append({"model": name, "error": str(e)})

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return results


# =========================================================================
# Loading saved classifiers
# =========================================================================

def load_classifier(model_dir):
    """Load a fine-tuned classifier from disk.

    Returns a dict with model, tokenizer, and num_labels.
    """
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()
    return {
        "model": model,
        "tokenizer": tokenizer,
        "num_labels": int(model.config.num_labels),
    }


def predict_classifier(model, tokenizer, texts, max_seq_len=512, batch_size=32):
    """Batched inference for a loaded classifier.

    Returns a dict with predictions (list of ints) and probabilities
    (list of list[float], one row per input).
    """
    texts = list(texts)
    if not texts:
        return {"predictions": [], "probabilities": []}

    device = _get_device()
    model = model.to(device)
    model.eval()

    all_preds = []
    all_probs = []
    for i in range(0, len(texts), int(batch_size)):
        batch = texts[i:i + int(batch_size)]
        inputs = tokenizer(
            batch, return_tensors="pt", truncation=True,
            padding=True, max_length=int(max_seq_len),
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        preds = np.argmax(probs, axis=1).tolist()
        all_preds.extend([int(p) for p in preds])
        all_probs.extend([[float(x) for x in row] for row in probs])

    try:
        model.cpu()
    except Exception:
        pass

    return {"predictions": all_preds, "probabilities": all_probs}


# =========================================================================
# Active Learning
# =========================================================================

def al_train(
    texts, labels, num_labels, model_name="ConfliBERT",
    epochs=3, batch_size=8, lr=2e-5, max_seq_len=512,
    dev_texts=None, dev_labels=None, task="binary",
    use_lora=False, lora_rank=8, lora_alpha=16, seed=42,
):
    """Train one active-learning round on the given labeled data.

    `seed` makes each round's training reproducible (see `finetune`).

    Returns a dict with keys: model, tokenizer, metrics, runtime.
    The model/tokenizer are raw Python objects kept on CPU; callers are
    expected to hold the reference across rounds.
    """
    from transformers import (
        AutoTokenizer,
        AutoModelForSequenceClassification,
        TrainingArguments,
        Trainer,
        set_seed,
    )

    set_seed(int(seed))

    model_id = MODEL_MAP.get(model_name, model_name)
    texts = list(texts)
    labels = [int(l) for l in labels]
    if task == "binary":
        num_labels = 2
    else:
        num_labels = int(num_labels)

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_id, num_labels=num_labels,
    )
    if use_lora:
        model = _apply_lora(model, lora_rank, lora_alpha)

    train_ds = _Dataset(
        tokenizer(texts, truncation=True, padding=True, max_length=int(max_seq_len)),
        labels,
    )
    dev_ds = None
    if dev_texts is not None and dev_labels is not None:
        dev_ds = _Dataset(
            tokenizer(
                list(dev_texts), truncation=True, padding=True,
                max_length=int(max_seq_len),
            ),
            [int(l) for l in dev_labels],
        )

    output_dir = tempfile.mkdtemp(prefix="conflibert_al_")
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=int(epochs),
        per_device_train_batch_size=int(batch_size),
        per_device_eval_batch_size=int(batch_size) * 2,
        learning_rate=float(lr),
        weight_decay=0.01,
        warmup_ratio=0.1,
        eval_strategy="epoch" if dev_ds is not None else "no",
        save_strategy="no",
        logging_steps=50,
        report_to="none",
        seed=int(seed),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        compute_metrics=_make_metrics_fn(task) if dev_ds is not None else None,
    )
    train_result = trainer.train()

    metrics = {}
    if dev_ds is not None:
        eval_results = trainer.evaluate()
        for k, v in eval_results.items():
            if isinstance(v, (int, float, np.floating)) and "epoch" not in k:
                metrics[k.replace("eval_", "")] = round(float(v), 4)

    runtime = round(train_result.metrics.get("train_runtime", 0), 1)

    trained = trainer.model
    if use_lora:
        trained = _merge_lora(trained)
    try:
        trained = trained.cpu()
    except Exception:
        pass
    trained.eval()

    return {
        "model": trained,
        "tokenizer": tokenizer,
        "metrics": metrics,
        "runtime": runtime,
    }


def al_score(
    model, tokenizer, texts, strategy="entropy",
    max_seq_len=512, batch_size=32,
):
    """Compute uncertainty scores for unlabeled texts. Higher = more uncertain.

    Strategies: 'entropy', 'margin', 'least_confidence'.
    Returns a list of floats, one per input text.
    """
    texts = list(texts)
    if not texts:
        return []

    device = _get_device()
    model = model.to(device)
    model.eval()

    scores = []
    for i in range(0, len(texts), int(batch_size)):
        batch = texts[i:i + int(batch_size)]
        inputs = tokenizer(
            batch, return_tensors="pt", truncation=True,
            padding=True, max_length=int(max_seq_len),
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=1).cpu().numpy()

        if strategy == "entropy":
            s = -np.sum(probs * np.log(probs + 1e-10), axis=1)
        elif strategy == "margin":
            sorted_p = np.sort(probs, axis=1)
            s = -(sorted_p[:, -1] - sorted_p[:, -2])
        elif strategy == "least_confidence":
            s = -np.max(probs, axis=1)
        else:
            raise ValueError(
                f"Unknown strategy '{strategy}'. "
                "Use 'entropy', 'margin', or 'least_confidence'."
            )
        scores.extend([float(x) for x in s])

    try:
        model.cpu()
    except Exception:
        pass
    return scores


def al_embed(model, tokenizer, texts, max_seq_len=512, batch_size=32):
    """Return [CLS] embeddings for each text as an (n, hidden) numpy array.

    Used by diversity-aware query selection to cluster candidates in
    representation space so the picked batch is not dominated by
    near-duplicates.
    """
    texts = list(texts)
    if not texts:
        return np.zeros((0, 0), dtype=np.float32)

    device = _get_device()
    model = model.to(device)
    model.eval()

    chunks = []
    for i in range(0, len(texts), int(batch_size)):
        batch = texts[i:i + int(batch_size)]
        inputs = tokenizer(
            batch, return_tensors="pt", truncation=True,
            padding=True, max_length=int(max_seq_len),
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            out = model(**inputs, output_hidden_states=True)
        last_hidden = out.hidden_states[-1]
        cls = last_hidden[:, 0, :].cpu().numpy().astype(np.float32)
        chunks.append(cls)

    try:
        model.cpu()
    except Exception:
        pass
    return np.vstack(chunks)


def al_save(model, tokenizer, save_dir):
    """Save the active-learning model and tokenizer to save_dir."""
    os.makedirs(save_dir, exist_ok=True)
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    return save_dir
