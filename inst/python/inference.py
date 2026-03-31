"""ConfliBERT inference module for the R package.

Models are loaded lazily and cached so the first call to each task
downloads and loads the model, and subsequent calls reuse it.
"""

import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import torch
import numpy as np
import gc
import tempfile

_models = {}

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
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
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
        try:
            from transformers import TFAutoModelForQuestionAnswering
        except ImportError:
            raise ImportError(
                "Question answering requires TensorFlow support in transformers. "
                "Install with: pip install 'transformers<5' tensorflow tf-keras"
            )
        name = "salsarra/ConfliBERT-QA"
        model = TFAutoModelForQuestionAnswering.from_pretrained(name)
        tokenizer = AutoTokenizer.from_pretrained(name)
    else:
        raise ValueError(f"Unknown task: {task}")

    _models[task] = (model, tokenizer)
    return model, tokenizer


def ner(text):
    """Run NER on a single text string.

    Returns a list of dicts with 'entity' and 'label' keys.
    """
    model, tokenizer = _load("ner")
    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)

    tag_ids = outputs.logits.argmax(dim=2).squeeze().tolist()
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"].squeeze().tolist())

    entities = []
    current_words = []
    current_label = None

    for i in range(len(tokens)):
        token = tokens[i]
        label = model.config.id2label[tag_ids[i]].split("-")[-1]

        if token.startswith("##"):
            if current_words:
                current_words[-1] += token[2:]
        elif label != "O":
            if label == current_label:
                current_words.append(token)
            else:
                if current_words:
                    entities.append({
                        "entity": " ".join(current_words),
                        "label": current_label,
                    })
                current_words = [token]
                current_label = label
        else:
            if current_words:
                entities.append({
                    "entity": " ".join(current_words),
                    "label": current_label,
                })
                current_words = []
                current_label = None

    if current_words:
        entities.append({
            "entity": " ".join(current_words),
            "label": current_label,
        })

    return entities


def classify(text):
    """Binary classification: conflict-related or not.

    Returns a dict with label, class, confidence, prob_negative, prob_positive.
    """
    model, tokenizer = _load("classify")
    device = _get_device()
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.softmax(outputs.logits, dim=1).squeeze()
    predicted = torch.argmax(probs).item()

    return {
        "label": "Positive" if predicted == 1 else "Negative",
        "class": int(predicted),
        "confidence": round(float(probs[predicted].item()), 4),
        "prob_negative": round(float(probs[0].item()), 4),
        "prob_positive": round(float(probs[1].item()), 4),
    }


def multilabel(text):
    """Multilabel classification across four event types.

    Returns a list of dicts with label, probability, predicted.
    """
    model, tokenizer = _load("multilabel")
    device = _get_device()
    categories = ["Armed Assault", "Bombing or Explosion", "Kidnapping", "Other"]

    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.sigmoid(outputs.logits).squeeze().tolist()

    return [
        {
            "label": categories[i],
            "probability": round(probs[i], 4),
            "predicted": bool(probs[i] >= 0.5),
        }
        for i in range(len(categories))
    ]


def qa(context, question):
    """Extractive question answering.

    Returns a dict with 'answer'.
    """
    import tensorflow as tf

    model, tokenizer = _load("qa")
    inputs = tokenizer(question, context, return_tensors="tf", truncation=True)
    outputs = model(inputs)

    start = tf.argmax(outputs.start_logits, axis=1).numpy()[0]
    end = tf.argmax(outputs.end_logits, axis=1).numpy()[0] + 1
    answer_tokens = tokenizer.convert_ids_to_tokens(
        inputs["input_ids"].numpy()[0][start:end]
    )
    answer = tokenizer.convert_tokens_to_string(answer_tokens)

    return {"answer": answer}


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
    predictions = [classify(t)["class"] for t in texts]

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
    save_dir=None,
):
    """Fine-tune a classification model and evaluate on the test set.

    Returns a dict with metrics, runtime, predictions, probabilities,
    true_labels, and model_dir.
    """
    from transformers import (
        AutoTokenizer,
        AutoModelForSequenceClassification,
        TrainingArguments,
        Trainer,
        EarlyStoppingCallback,
    )

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
        seed=42,
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

    if save_dir:
        trainer.model.save_pretrained(save_dir)
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
):
    """Fine-tune multiple models on the same data and compare performance.

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
