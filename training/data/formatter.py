"""Format synthesized data into HuggingFace Dataset for training."""

import json
import re
from pathlib import Path
from datasets import Dataset, Audio, DatasetDict


def extract_tags(tagged_text: str) -> list[dict]:
    """Extract tags and their positions from tagged text."""
    tags = []
    for match in re.finditer(r'\[([^\]]+)\]', tagged_text):
        tags.append({
            "tag": match.group(1),
            "start_char": match.start(),
            "end_char": match.end(),
        })
    return tags


def strip_tags(tagged_text: str) -> str:
    """Remove all [tags] from text, leaving plain transcription."""
    return re.sub(r'\[[^\]]+\]\s*', '', tagged_text).strip()


def has_emphasis(text: str) -> bool:
    """Check if text contains CAPS emphasis (2+ consecutive uppercase letters forming a word)."""
    return bool(re.search(r'\b[A-Z]{2,}\b', text))


def format_dataset(
    manifest_path: str = "data/audio/manifest.json",
    output_dir: str = "data/processed",
) -> DatasetDict:
    """Convert manifest + audio files into a HuggingFace Dataset."""
    with open(manifest_path) as f:
        manifest = json.load(f)

    records = [m for m in manifest if m.get("synthesis_success", True)]

    rows = []
    for record in records:
        tagged_text = record["tagged_text"]
        plain_text = record.get("plain_text", strip_tags(tagged_text))
        tags = extract_tags(tagged_text)

        rows.append({
            "audio": record["audio_path"],
            "tagged_text": tagged_text,
            "plain_text": plain_text,
            "tags": json.dumps(tags),
            "tag_count": len(tags),
            "has_emphasis": has_emphasis(tagged_text),
            "slice_type": record.get("slice_type", "unknown"),
            "domain": record.get("domain", "unknown"),
            "voice_id": record.get("voice_id", "unknown"),
        })

    ds = Dataset.from_dict({k: [r[k] for r in rows] for k in rows[0]})
    ds = ds.cast_column("audio", Audio(sampling_rate=16000))

    # Train/val/test split: 80/10/10
    ds_split = ds.train_test_split(test_size=0.2, seed=42)
    val_test = ds_split["test"].train_test_split(test_size=0.5, seed=42)

    final = DatasetDict({
        "train": ds_split["train"],
        "validation": val_test["train"],
        "test": val_test["test"],
    })

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    final.save_to_disk(output_dir)
    print(f"Dataset saved to {output_dir}")
    print(f"  Train: {len(final['train'])}, Val: {len(final['validation'])}, Test: {len(final['test'])}")

    return final


if __name__ == "__main__":
    format_dataset()
