"""Tag-level metrics for Evoxtral evaluation."""

import re
from collections import Counter


def extract_tags(text: str) -> list[str]:
    """Extract all [tag] names from text."""
    return [m.group(1).lower() for m in re.finditer(r'\[([^\]]+)\]', text)]


def extract_emphasis_words(text: str) -> list[str]:
    """Extract CAPS-emphasized words."""
    return [m.group(0).lower() for m in re.finditer(r'\b[A-Z]{2,}\b', text)]


def strip_tags(text: str) -> str:
    """Remove all [tags] and normalize CAPS to lowercase."""
    text = re.sub(r'\[[^\]]+\]\s*', '', text)
    # Convert CAPS words to lowercase
    text = re.sub(r'\b[A-Z]{2,}\b', lambda m: m.group(0).lower(), text)
    return text.strip()


def tag_f1(predicted: str, reference: str) -> dict:
    """Compute precision, recall, F1 for tag extraction."""
    pred_tags = Counter(extract_tags(predicted))
    ref_tags = Counter(extract_tags(reference))

    if not ref_tags and not pred_tags:
        return {"tag_precision": 1.0, "tag_recall": 1.0, "tag_f1": 1.0}
    if not ref_tags:
        return {"tag_precision": 0.0, "tag_recall": 1.0, "tag_f1": 0.0}
    if not pred_tags:
        return {"tag_precision": 1.0, "tag_recall": 0.0, "tag_f1": 0.0}

    # Intersection using min counts
    common = sum((pred_tags & ref_tags).values())
    precision = common / sum(pred_tags.values()) if pred_tags else 0
    recall = common / sum(ref_tags.values()) if ref_tags else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {"tag_precision": precision, "tag_recall": recall, "tag_f1": f1}


def emphasis_f1(predicted: str, reference: str) -> dict:
    """Compute F1 for CAPS emphasis words."""
    pred_words = Counter(extract_emphasis_words(predicted))
    ref_words = Counter(extract_emphasis_words(reference))

    if not ref_words and not pred_words:
        return {"emphasis_precision": 1.0, "emphasis_recall": 1.0, "emphasis_f1": 1.0}
    if not ref_words:
        return {"emphasis_precision": 0.0, "emphasis_recall": 1.0, "emphasis_f1": 0.0}
    if not pred_words:
        return {"emphasis_precision": 1.0, "emphasis_recall": 0.0, "emphasis_f1": 0.0}

    common = sum((pred_words & ref_words).values())
    precision = common / sum(pred_words.values()) if pred_words else 0
    recall = common / sum(ref_words.values()) if ref_words else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {"emphasis_precision": precision, "emphasis_recall": recall, "emphasis_f1": f1}


def tag_hallucination_rate(predicted: str, reference: str) -> float:
    """Rate of tags in prediction that don't exist in reference (hallucinated tags)."""
    pred_tags = extract_tags(predicted)
    ref_tags = set(extract_tags(reference))

    if not pred_tags:
        return 0.0

    hallucinated = sum(1 for t in pred_tags if t not in ref_tags)
    return hallucinated / len(pred_tags)
