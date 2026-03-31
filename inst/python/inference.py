"""ConfliBERT inference module for the R package.

Models are loaded lazily and cached so the first call to each task
downloads and loads the model, and subsequent calls reuse it.
"""

import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import torch

_models = {}


def _get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _load(task):
    if task in _models:
        return _models[task]

    from transformers import (
        AutoTokenizer,
        AutoModelForSequenceClassification,
        AutoModelForTokenClassification,
        TFAutoModelForQuestionAnswering,
    )

    device = _get_device()

    if task == "ner":
        name = "eventdata-utd/conflibert-named-entity-recognition"
        model = AutoModelForTokenClassification.from_pretrained(name).to(device)
        tokenizer = AutoTokenizer.from_pretrained(name)
    elif task == "classify":
        name = "eventdata-utd/conflibert-binary-classification"
        model = AutoModelForSequenceClassification.from_pretrained(name).to(device)
        tokenizer = AutoTokenizer.from_pretrained(name)
    elif task == "multilabel":
        name = "eventdata-utd/conflibert-satp-relevant-multilabel"
        model = AutoModelForSequenceClassification.from_pretrained(name).to(device)
        tokenizer = AutoTokenizer.from_pretrained(name)
    elif task == "qa":
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
