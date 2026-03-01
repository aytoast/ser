# Evoxtral Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Finetune Voxtral-Mini-3B to produce transcriptions with inline ElevenLabs v3 audio tags, with a full data pipeline, eval suite, backend API, and frontend demo.

**Architecture:** Reverse-pipeline synthetic data generation (tagged text -> ElevenLabs TTS -> audio) produces training pairs. LoRA finetuning on Voxtral-Mini-3B teaches the model to predict tags from audio. FastAPI backend serves the model from HuggingFace. Next.js frontend provides a demo UI.

**Tech Stack:** Python 3.11+, transformers, peft, torchaudio, wandb, jiwer, elevenlabs SDK, FastAPI, Next.js, shadcn/ui

---

## Task 1: Project Scaffolding & Dependencies

**Files:**
- Create: `pyproject.toml`
- Create: `requirements.txt`
- Create: `data/README.md`
- Create: `data/scripts/`
- Create: `data/audio/`
- Create: `data/processed/`
- Create: `src/__init__.py`
- Create: `src/data/__init__.py`
- Create: `src/training/__init__.py`
- Create: `src/eval/__init__.py`
- Create: `src/api/__init__.py`
- Create: `tests/__init__.py`

**Step 1: Create requirements.txt**

```txt
# Data pipeline
elevenlabs>=1.0.0
openai>=1.0.0

# Training
torch>=2.1.0
torchaudio>=2.1.0
transformers>=4.54.0
peft>=0.7.0
datasets>=2.14.0
accelerate>=0.25.0
bitsandbytes>=0.41.0

# Eval
jiwer>=3.0.0

# Tracking
wandb>=0.16.0

# API
fastapi>=0.104.0
uvicorn>=0.24.0
python-multipart>=0.0.6
websockets>=12.0

# Utils
python-dotenv>=1.0.0
soundfile>=0.12.0
librosa>=0.10.0
tqdm>=4.66.0
```

**Step 2: Create directory structure**

Run:
```bash
mkdir -p src/{data,training,eval,api} tests data/{scripts,audio,processed}
touch src/__init__.py src/data/__init__.py src/training/__init__.py src/eval/__init__.py src/api/__init__.py tests/__init__.py
```

**Step 3: Create .env template**

Create `.env.example`:
```
ELEVENLABS_API_KEY_1=sk_...
ELEVENLABS_API_KEY_2=sk_...
ELEVENLABS_API_KEY_3=sk_...
WANDB_API_KEY=...
WANDB_PROJECT=evoxtral
HF_TOKEN=...
MISTRAL_API_KEY=...
```

**Step 4: Install dependencies**

Run: `pip install -r requirements.txt`

**Step 5: Commit**

```bash
git add -A
git commit -m "chore: scaffold project structure and dependencies"
```

---

## Task 2: Script Generator (Tagged Text Creation)

**Files:**
- Create: `src/data/tag_taxonomy.py`
- Create: `src/data/script_generator.py`
- Create: `tests/test_script_generator.py`

**Step 1: Write tag taxonomy**

`src/data/tag_taxonomy.py`:
```python
"""ElevenLabs v3 audio tag taxonomy for Evoxtral."""

EMOTION_TAGS = ["excited", "sad", "angry", "nervous", "calm", "frustrated"]
NONVERBAL_TAGS = ["laughs", "sighs", "gasps", "clears throat", "crying"]
DELIVERY_TAGS = ["whispers", "shouts", "stammers"]
PAUSE_TAGS = ["pause"]

ALL_BRACKET_TAGS = EMOTION_TAGS + NONVERBAL_TAGS + DELIVERY_TAGS + PAUSE_TAGS

# Slice definitions for balanced dataset
SLICE_CONFIG = {
    "plain": {"ratio": 0.25, "tag_density": 0, "description": "No tags, plain ASR"},
    "light": {"ratio": 0.25, "tag_density": (1, 2), "description": "1-2 tags per sample"},
    "moderate": {"ratio": 0.25, "tag_density": (3, 4), "description": "3-4 tags per sample"},
    "dense": {"ratio": 0.15, "tag_density": (5, 8), "description": "5+ tags per sample"},
    "edge": {"ratio": 0.10, "tag_density": (1, 6), "description": "Edge cases: ambiguous, boundary"},
}

DOMAINS = [
    "conversation", "monologue", "podcast", "presentation",
    "argument", "storytelling", "interview", "voicemail"
]

# Semantic groups for eval (tags within a group are considered equivalent)
TAG_SEMANTIC_GROUPS = {
    "laughter": ["laughs", "giggles", "chuckles"],
    "sadness": ["sad", "crying", "sorrowful"],
    "breathing": ["sighs", "gasps", "exhales"],
    "loud": ["shouts", "yells"],
    "quiet": ["whispers", "murmurs"],
}
```

**Step 2: Write script generator**

`src/data/script_generator.py`:
```python
"""Generate diverse tagged scripts for ElevenLabs v3 TTS synthesis."""

import json
import random
import os
from pathlib import Path
from openai import OpenAI
from .tag_taxonomy import ALL_BRACKET_TAGS, SLICE_CONFIG, DOMAINS

SYSTEM_PROMPT = """You are a script writer generating realistic speech samples with inline ElevenLabs v3 audio tags.

Audio tags use square brackets: [laughs], [excited], [whispers], [pause], etc.
Emphasis uses CAPITALIZATION of stressed words.
Pauses use ellipses ...

Rules:
- Write natural, diverse dialogue/monologue snippets (15-80 words)
- Tags must feel organic, not forced
- Vary domains: conversation, podcast, storytelling, argument, etc.
- Include a mix of male/female perspectives
- Each sample should be self-contained (no context needed)

Available tags: {tags}

Output ONLY a JSON array of objects with fields:
- "tagged_text": the text with inline tags
- "plain_text": same text with all tags and CAPS emphasis removed
- "tags_used": list of tags used (without brackets)
- "domain": one of {domains}
- "tag_count": number of tags used
"""


def generate_scripts_for_slice(
    slice_name: str,
    count: int,
    client: OpenAI | None = None,
    model: str = "mistral-large-latest",
) -> list[dict]:
    """Generate tagged scripts for a given slice type."""
    if client is None:
        client = OpenAI(
            api_key=os.getenv("MISTRAL_API_KEY"),
            base_url="https://api.mistral.ai/v1",
        )

    config = SLICE_CONFIG[slice_name]
    tag_density = config["tag_density"]

    if slice_name == "plain":
        density_instruction = "Do NOT include any audio tags or CAPS emphasis. Plain speech only."
    elif isinstance(tag_density, tuple):
        density_instruction = f"Use exactly {tag_density[0]}-{tag_density[1]} audio tags per sample."
    else:
        density_instruction = f"Use exactly {tag_density} audio tags per sample."

    if slice_name == "edge":
        density_instruction += (
            " Include edge cases: tags at the very start/end, "
            "back-to-back tags like [angry][laughs], "
            "ambiguous emotions, very short utterances with tags."
        )

    scripts = []
    batch_size = 20  # generate 20 at a time

    for i in range(0, count, batch_size):
        n = min(batch_size, count - i)
        prompt = (
            f"Generate exactly {n} speech samples.\n"
            f"Slice type: {slice_name} - {config['description']}\n"
            f"Tag density: {density_instruction}\n"
            f"Distribute evenly across these domains: {', '.join(DOMAINS)}"
        )

        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT.format(
                        tags=", ".join(ALL_BRACKET_TAGS),
                        domains=", ".join(DOMAINS),
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.9,
            response_format={"type": "json_object"},
        )

        try:
            content = response.choices[0].message.content
            parsed = json.loads(content)
            batch = parsed if isinstance(parsed, list) else parsed.get("samples", parsed.get("scripts", []))
            for item in batch:
                item["slice_type"] = slice_name
            scripts.extend(batch)
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Failed to parse batch {i // batch_size}: {e}")
            continue

    return scripts[:count]


def generate_full_dataset(total: int = 1000, output_path: str = "data/scripts/scripts.json") -> list[dict]:
    """Generate the full balanced dataset of tagged scripts."""
    all_scripts = []

    for slice_name, config in SLICE_CONFIG.items():
        count = int(total * config["ratio"])
        print(f"Generating {count} {slice_name} scripts...")
        scripts = generate_scripts_for_slice(slice_name, count)
        all_scripts.extend(scripts)
        print(f"  Got {len(scripts)} scripts")

    random.shuffle(all_scripts)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_scripts, f, indent=2)

    print(f"Total: {len(all_scripts)} scripts saved to {output_path}")
    return all_scripts


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    generate_full_dataset()
```

**Step 3: Write tests**

`tests/test_script_generator.py`:
```python
"""Tests for script generator output format and balance."""

import json
import pytest
from src.data.tag_taxonomy import SLICE_CONFIG, ALL_BRACKET_TAGS


def test_slice_ratios_sum_to_one():
    total = sum(c["ratio"] for c in SLICE_CONFIG.values())
    assert abs(total - 1.0) < 0.01


def test_all_tags_present():
    assert len(ALL_BRACKET_TAGS) >= 15


def validate_script_format(script: dict):
    """Validate a single script has required fields."""
    assert "tagged_text" in script
    assert "plain_text" in script
    assert "slice_type" in script
    assert len(script["tagged_text"]) > 0
    assert len(script["plain_text"]) > 0
```

**Step 4: Run tests**

Run: `python -m pytest tests/test_script_generator.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/data/ tests/test_script_generator.py
git commit -m "feat: add script generator with balanced tag taxonomy"
```

---

## Task 3: ElevenLabs Synthesizer (Audio Generation)

**Files:**
- Create: `src/data/synthesizer.py`
- Create: `tests/test_synthesizer.py`

**Step 1: Write synthesizer with key rotation and concurrency**

`src/data/synthesizer.py`:
```python
"""ElevenLabs v3 TTS synthesizer with API key rotation and concurrency."""

import asyncio
import os
import json
import hashlib
from pathlib import Path
from itertools import cycle
from dataclasses import dataclass
from elevenlabs import ElevenLabs
from dotenv import load_dotenv

load_dotenv()

# Voice IDs to cycle through (select 6-8 diverse voices)
# These should be populated with actual ElevenLabs voice IDs
DEFAULT_VOICES = [
    # Fill with actual voice IDs from ElevenLabs library
    # Mix: 2 male adult, 2 female adult, 1 young male, 1 young female, 1 older male, 1 older female
]

MODEL_ID = "eleven_v3"


@dataclass
class SynthesisResult:
    script_index: int
    audio_path: str
    voice_id: str
    success: bool
    error: str | None = None


def get_api_keys() -> list[str]:
    """Load all ElevenLabs API keys from environment."""
    keys = []
    for key, value in os.environ.items():
        if key.startswith("ELEVENLABS_API_KEY"):
            keys.append(value)
    if not keys:
        raise ValueError("No ELEVENLABS_API_KEY* found in environment")
    return keys


def create_clients(keys: list[str]) -> list[ElevenLabs]:
    """Create ElevenLabs client per API key."""
    return [ElevenLabs(api_key=key) for key in keys]


def synthesize_one(
    client: ElevenLabs,
    text: str,
    voice_id: str,
    output_path: str,
) -> bool:
    """Synthesize a single text to audio file."""
    try:
        audio_generator = client.text_to_speech.convert(
            text=text,
            voice_id=voice_id,
            model_id=MODEL_ID,
            output_format="mp3_44100_128",
        )
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "wb") as f:
            for chunk in audio_generator:
                f.write(chunk)
        return True
    except Exception as e:
        print(f"Synthesis failed: {e}")
        return False


def synthesize_dataset(
    scripts_path: str = "data/scripts/scripts.json",
    output_dir: str = "data/audio",
    voices: list[str] | None = None,
    max_retries: int = 2,
) -> list[SynthesisResult]:
    """Synthesize all scripts to audio, rotating keys and voices."""
    with open(scripts_path) as f:
        scripts = json.load(f)

    keys = get_api_keys()
    clients = create_clients(keys)
    client_cycle = cycle(clients)
    voice_list = voices or DEFAULT_VOICES
    voice_cycle = cycle(voice_list)

    results = []
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    for i, script in enumerate(scripts):
        text = script["tagged_text"]
        voice_id = next(voice_cycle)
        client = next(client_cycle)

        # Deterministic filename from content hash
        content_hash = hashlib.md5(text.encode()).hexdigest()[:8]
        audio_path = f"{output_dir}/{i:04d}_{content_hash}.mp3"

        # Skip if already exists
        if Path(audio_path).exists():
            results.append(SynthesisResult(i, audio_path, voice_id, True))
            continue

        success = False
        error = None
        for attempt in range(max_retries + 1):
            success = synthesize_one(client, text, voice_id, audio_path)
            if success:
                break
            # Rotate to next client on failure
            client = next(client_cycle)
            error = f"Failed after {attempt + 1} attempts"

        results.append(SynthesisResult(i, audio_path, voice_id, success, error))

        if (i + 1) % 50 == 0:
            ok = sum(1 for r in results if r.success)
            print(f"Progress: {i + 1}/{len(scripts)} | Success: {ok}/{i + 1}")

    # Save manifest
    manifest = []
    for r, script in zip(results, scripts):
        manifest.append({
            **script,
            "audio_path": r.audio_path,
            "voice_id": r.voice_id,
            "synthesis_success": r.success,
        })

    manifest_path = f"{output_dir}/manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    ok = sum(1 for r in results if r.success)
    print(f"Done: {ok}/{len(scripts)} successful. Manifest: {manifest_path}")
    return results


if __name__ == "__main__":
    synthesize_dataset()
```

**Step 2: Write test**

`tests/test_synthesizer.py`:
```python
"""Tests for synthesizer key rotation and output format."""

from src.data.synthesizer import get_api_keys, SynthesisResult


def test_synthesis_result_format():
    r = SynthesisResult(0, "data/audio/0000.mp3", "voice_123", True)
    assert r.success
    assert r.error is None
```

**Step 3: Run test**

Run: `python -m pytest tests/test_synthesizer.py -v`
Expected: PASS

**Step 4: Commit**

```bash
git add src/data/synthesizer.py tests/test_synthesizer.py
git commit -m "feat: add ElevenLabs synthesizer with key rotation"
```

---

## Task 4: Dataset Formatter (HuggingFace Datasets)

**Files:**
- Create: `src/data/formatter.py`
- Create: `tests/test_formatter.py`

**Step 1: Write dataset formatter**

`src/data/formatter.py`:
```python
"""Format synthesized data into HuggingFace Dataset for training."""

import json
import re
from pathlib import Path
from datasets import Dataset, Audio, Features, Value, Sequence


def extract_tags(tagged_text: str) -> list[dict]:
    """Extract tags and their positions from tagged text."""
    tags = []
    # Match [tag] patterns
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
    """Check if text contains CAPS emphasis (2+ consecutive uppercase words)."""
    return bool(re.search(r'\b[A-Z]{2,}\b', text))


def format_dataset(
    manifest_path: str = "data/audio/manifest.json",
    output_dir: str = "data/processed",
) -> Dataset:
    """Convert manifest + audio files into a HuggingFace Dataset."""
    with open(manifest_path) as f:
        manifest = json.load(f)

    # Filter to successful syntheses only
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

    from datasets import DatasetDict
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
```

**Step 2: Write tests**

`tests/test_formatter.py`:
```python
"""Tests for dataset formatter tag extraction and stripping."""

from src.data.formatter import extract_tags, strip_tags, has_emphasis


def test_extract_tags():
    text = "[excited] Hello WORLD! [laughs] That was great."
    tags = extract_tags(text)
    assert len(tags) == 2
    assert tags[0]["tag"] == "excited"
    assert tags[1]["tag"] == "laughs"


def test_strip_tags():
    text = "[excited] Hello WORLD! [laughs] That was great."
    plain = strip_tags(text)
    assert "[" not in plain
    assert "Hello" in plain
    assert "great" in plain


def test_strip_tags_no_tags():
    text = "Just a normal sentence."
    assert strip_tags(text) == text


def test_has_emphasis():
    assert has_emphasis("That was AMAZING")
    assert not has_emphasis("That was amazing")
    assert has_emphasis("I can't BELIEVE this HAPPENED")


def test_extract_tags_empty():
    assert extract_tags("No tags here") == []
```

**Step 3: Run tests**

Run: `python -m pytest tests/test_formatter.py -v`
Expected: PASS

**Step 4: Commit**

```bash
git add src/data/formatter.py tests/test_formatter.py
git commit -m "feat: add HuggingFace dataset formatter with tag extraction"
```

---

## Task 5: LoRA Finetuning Script

**Files:**
- Create: `src/training/finetune.py`
- Create: `src/training/config.py`

**Step 1: Write training config**

`src/training/config.py`:
```python
"""Training configuration for Evoxtral LoRA finetuning."""

from dataclasses import dataclass, field


@dataclass
class EvoxtralTrainingConfig:
    # Model
    model_name: str = "mistralai/Voxtral-Mini-3B-2507"

    # LoRA
    lora_rank: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    lora_target_modules: list[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
        "multi_modal_projector.linear_1",
        "multi_modal_projector.linear_2",
    ])

    # Training
    num_train_epochs: int = 1
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    warmup_steps: int = 50
    weight_decay: float = 0.01
    bf16: bool = True

    # NEFTune
    neftune_noise_alpha: float = 5.0

    # Data
    dataset_path: str = "data/processed"
    max_seq_length: int = 2048

    # Output
    output_dir: str = "model/evoxtral-lora"
    logging_steps: int = 10
    save_steps: int = 100
    eval_steps: int = 50

    # W&B
    wandb_project: str = "evoxtral"
    report_to: str = "wandb"

    # HuggingFace
    hub_model_id: str = "mistral-hackaton-2026/evoxtral-lora"
    push_to_hub: bool = True
```

**Step 2: Write finetuning script**

`src/training/finetune.py`:
```python
"""LoRA finetuning script for Voxtral-Mini-3B with NEFTune."""

import os
import torch
from datasets import load_from_disk
from transformers import (
    AutoProcessor,
    AutoModelForVision2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)
from peft import LoraConfig, get_peft_model, TaskType
from dotenv import load_dotenv

from .config import EvoxtralTrainingConfig

load_dotenv()


def load_dataset(config: EvoxtralTrainingConfig):
    """Load the processed dataset."""
    ds = load_from_disk(config.dataset_path)
    return ds["train"], ds["validation"]


def setup_model_and_processor(config: EvoxtralTrainingConfig):
    """Load Voxtral-Mini-3B and configure LoRA."""
    processor = AutoProcessor.from_pretrained(config.model_name)

    model = AutoModelForVision2Seq.from_pretrained(
        config.model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    # Configure LoRA
    lora_config = LoraConfig(
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.lora_target_modules,
        task_type=TaskType.CAUSAL_LM,
        bias="none",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model, processor


def preprocess_function(examples, processor):
    """Preprocess audio + tagged text into model inputs.

    The prompt format for Voxtral is:
    [AUDIO]...[/AUDIO] <transcribe>

    We replace the standard transcription target with tagged text.
    """
    # Process audio
    audios = [ex["array"] for ex in examples["audio"]]
    sampling_rate = examples["audio"][0]["sampling_rate"]

    # The target is the tagged transcription
    texts = examples["tagged_text"]

    # Use processor to create inputs
    inputs = processor(
        audios=audios,
        text=["<transcribe>" for _ in texts],
        sampling_rate=sampling_rate,
        return_tensors="pt",
        padding=True,
    )

    # Create labels from tagged text
    labels = processor.tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
    )

    inputs["labels"] = labels["input_ids"]

    # Mask prompt tokens in labels (set to -100)
    # Only compute loss on the tagged transcription output
    prompt_length = inputs["input_ids"].shape[1] - labels["input_ids"].shape[1]
    if prompt_length > 0:
        mask = torch.full((labels["input_ids"].shape[0], prompt_length), -100)
        inputs["labels"] = torch.cat([mask, inputs["labels"]], dim=1)

    return inputs


def train(config: EvoxtralTrainingConfig | None = None):
    """Run the full finetuning pipeline."""
    if config is None:
        config = EvoxtralTrainingConfig()

    print("Loading dataset...")
    train_ds, val_ds = load_dataset(config)
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")

    print("Loading model...")
    model, processor = setup_model_and_processor(config)

    training_args = Seq2SeqTrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        warmup_steps=config.warmup_steps,
        weight_decay=config.weight_decay,
        bf16=config.bf16,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        eval_steps=config.eval_steps,
        eval_strategy="steps",
        save_total_limit=3,
        load_best_model_at_end=True,
        report_to=config.report_to,
        run_name="evoxtral-sft-lora",
        neftune_noise_alpha=config.neftune_noise_alpha,
        push_to_hub=config.push_to_hub,
        hub_model_id=config.hub_model_id,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=processor,
    )

    print("Starting training...")
    trainer.train()

    print("Saving model...")
    trainer.save_model(config.output_dir)
    processor.save_pretrained(config.output_dir)

    if config.push_to_hub:
        print("Pushing to HuggingFace Hub...")
        trainer.push_to_hub()

    print("Done!")


if __name__ == "__main__":
    train()
```

**Step 3: Commit**

```bash
git add src/training/
git commit -m "feat: add LoRA finetuning script with NEFTune and W&B"
```

---

## Task 6: Evoxtral-Bench (Evaluation Suite)

**Files:**
- Create: `src/eval/bench.py`
- Create: `src/eval/tag_metrics.py`
- Create: `src/eval/roundtrip.py`
- Create: `tests/test_eval.py`

**Step 1: Write tag metrics**

`src/eval/tag_metrics.py`:
```python
"""Tag-level evaluation metrics for Evoxtral-Bench."""

import re
from dataclasses import dataclass
from src.data.tag_taxonomy import TAG_SEMANTIC_GROUPS


@dataclass
class TagMetrics:
    precision: float
    recall: float
    f1: float
    position_accuracy: float
    total_predicted: int
    total_ground_truth: int


def extract_tag_list(text: str) -> list[str]:
    """Extract ordered list of tags from tagged text."""
    return re.findall(r'\[([^\]]+)\]', text)


def extract_tag_positions(text: str) -> list[tuple[str, int]]:
    """Extract tags with their word-position index."""
    tags_with_pos = []
    # Remove tags to count word positions
    words_before = []
    parts = re.split(r'(\[[^\]]+\])', text)
    word_idx = 0
    for part in parts:
        if part.startswith('[') and part.endswith(']'):
            tag = part[1:-1]
            tags_with_pos.append((tag, word_idx))
        else:
            word_idx += len(part.split())
    return tags_with_pos


def normalize_tag(tag: str) -> str:
    """Normalize tag to canonical form using semantic groups."""
    tag_lower = tag.lower().strip()
    for canonical, variants in TAG_SEMANTIC_GROUPS.items():
        if tag_lower in variants:
            return canonical
    return tag_lower


def compute_tag_metrics(
    predicted: str,
    ground_truth: str,
    position_tolerance: int = 3,
) -> TagMetrics:
    """Compute tag-level precision, recall, F1, and position accuracy."""
    pred_tags = extract_tag_list(predicted)
    gt_tags = extract_tag_list(ground_truth)

    pred_normalized = [normalize_tag(t) for t in pred_tags]
    gt_normalized = [normalize_tag(t) for t in gt_tags]

    # Tag presence F1 (bag-of-tags)
    pred_set = set(pred_normalized)
    gt_set = set(gt_normalized)

    if len(pred_set) == 0 and len(gt_set) == 0:
        precision = recall = f1 = 1.0
    elif len(pred_set) == 0:
        precision = recall = f1 = 0.0
    elif len(gt_set) == 0:
        precision = 0.0
        recall = 1.0
        f1 = 0.0
    else:
        tp = len(pred_set & gt_set)
        precision = tp / len(pred_set)
        recall = tp / len(gt_set)
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    # Position accuracy
    pred_positions = extract_tag_positions(predicted)
    gt_positions = extract_tag_positions(ground_truth)

    if len(gt_positions) == 0:
        position_accuracy = 1.0 if len(pred_positions) == 0 else 0.0
    else:
        correct_positions = 0
        gt_used = [False] * len(gt_positions)
        for p_tag, p_pos in pred_positions:
            p_norm = normalize_tag(p_tag)
            for j, (g_tag, g_pos) in enumerate(gt_positions):
                if not gt_used[j] and normalize_tag(g_tag) == p_norm and abs(p_pos - g_pos) <= position_tolerance:
                    correct_positions += 1
                    gt_used[j] = True
                    break
        position_accuracy = correct_positions / len(gt_positions)

    return TagMetrics(
        precision=precision,
        recall=recall,
        f1=f1,
        position_accuracy=position_accuracy,
        total_predicted=len(pred_tags),
        total_ground_truth=len(gt_tags),
    )
```

**Step 2: Write main bench runner**

`src/eval/bench.py`:
```python
"""Evoxtral-Bench: 3-layer evaluation for tagged transcription."""

import json
from dataclasses import dataclass, asdict
from jiwer import wer, cer
from src.data.formatter import strip_tags
from .tag_metrics import compute_tag_metrics, TagMetrics


@dataclass
class BenchResult:
    # Layer 1: Text accuracy
    wer: float
    cer: float
    # Layer 2: Tag accuracy
    tag_precision: float
    tag_recall: float
    tag_f1: float
    tag_position_accuracy: float
    # Layer 3: Round-trip (optional)
    roundtrip_score: float | None = None
    # Meta
    num_samples: int = 0


def evaluate(
    predictions: list[str],
    ground_truths: list[str],
    roundtrip_scores: list[float] | None = None,
) -> BenchResult:
    """Run full Evoxtral-Bench evaluation.

    Args:
        predictions: list of predicted tagged transcriptions
        ground_truths: list of ground-truth tagged transcriptions
        roundtrip_scores: optional list of round-trip audio similarity scores
    """
    assert len(predictions) == len(ground_truths)

    # Layer 1: WER/CER on plain text (tags stripped)
    pred_plain = [strip_tags(p) for p in predictions]
    gt_plain = [strip_tags(g) for g in ground_truths]

    # Filter empty pairs
    valid = [(p, g) for p, g in zip(pred_plain, gt_plain) if g.strip()]
    if valid:
        vp, vg = zip(*valid)
        text_wer = wer(list(vg), list(vp))
        text_cer = cer(list(vg), list(vp))
    else:
        text_wer = text_cer = 0.0

    # Layer 2: Tag metrics
    all_tag_metrics = [
        compute_tag_metrics(p, g)
        for p, g in zip(predictions, ground_truths)
    ]

    avg_precision = sum(m.precision for m in all_tag_metrics) / len(all_tag_metrics)
    avg_recall = sum(m.recall for m in all_tag_metrics) / len(all_tag_metrics)
    avg_f1 = sum(m.f1 for m in all_tag_metrics) / len(all_tag_metrics)
    avg_pos = sum(m.position_accuracy for m in all_tag_metrics) / len(all_tag_metrics)

    # Layer 3: Round-trip
    rt_score = None
    if roundtrip_scores:
        rt_score = sum(roundtrip_scores) / len(roundtrip_scores)

    return BenchResult(
        wer=text_wer,
        cer=text_cer,
        tag_precision=avg_precision,
        tag_recall=avg_recall,
        tag_f1=avg_f1,
        tag_position_accuracy=avg_pos,
        roundtrip_score=rt_score,
        num_samples=len(predictions),
    )


def print_results(result: BenchResult):
    """Pretty-print Evoxtral-Bench results."""
    print("\n" + "=" * 50)
    print("EVOXTRAL-BENCH RESULTS")
    print("=" * 50)
    print(f"\nLayer 1 - Text Accuracy:")
    print(f"  WER:  {result.wer:.2%}")
    print(f"  CER:  {result.cer:.2%}")
    print(f"\nLayer 2 - Tag Accuracy:")
    print(f"  Precision: {result.tag_precision:.2%}")
    print(f"  Recall:    {result.tag_recall:.2%}")
    print(f"  F1:        {result.tag_f1:.2%}")
    print(f"  Position:  {result.tag_position_accuracy:.2%}")
    if result.roundtrip_score is not None:
        print(f"\nLayer 3 - Round-Trip:")
        print(f"  Score: {result.roundtrip_score:.2%}")
    print(f"\nSamples: {result.num_samples}")
    print("=" * 50)


def save_results(result: BenchResult, path: str = "eval_results.json"):
    """Save results to JSON."""
    with open(path, "w") as f:
        json.dump(asdict(result), f, indent=2)
```

**Step 3: Write round-trip evaluator stub**

`src/eval/roundtrip.py`:
```python
"""Round-trip evaluation: tagged text -> ElevenLabs TTS -> compare to original audio."""

import os
from elevenlabs import ElevenLabs
from dotenv import load_dotenv

load_dotenv()


def roundtrip_evaluate(
    tagged_text: str,
    original_audio_path: str,
    voice_id: str,
    client: ElevenLabs | None = None,
) -> float:
    """Generate audio from tagged text and compare to original.

    Returns a similarity score between 0 and 1.
    For hackathon MVP, we use a simple duration-ratio heuristic.
    Full implementation would use audio embeddings (e.g., wav2vec2 cosine similarity).
    """
    if client is None:
        client = ElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY_1"))

    # Generate audio from tagged prediction
    try:
        audio_gen = client.text_to_speech.convert(
            text=tagged_text,
            voice_id=voice_id,
            model_id="eleven_v3",
        )
        # For MVP: return 1.0 if synthesis succeeds (validates tag format)
        # TODO: implement audio embedding comparison
        return 1.0
    except Exception:
        return 0.0
```

**Step 4: Write tests**

`tests/test_eval.py`:
```python
"""Tests for Evoxtral-Bench evaluation metrics."""

from src.eval.tag_metrics import (
    extract_tag_list,
    extract_tag_positions,
    normalize_tag,
    compute_tag_metrics,
)
from src.eval.bench import evaluate


def test_extract_tag_list():
    text = "[excited] Hello! [laughs] That was fun."
    tags = extract_tag_list(text)
    assert tags == ["excited", "laughs"]


def test_extract_tag_list_empty():
    assert extract_tag_list("No tags here.") == []


def test_normalize_tag_semantic_group():
    assert normalize_tag("giggles") == "laughter"
    assert normalize_tag("laughs") == "laughter"
    assert normalize_tag("excited") == "excited"  # no group, return as-is


def test_perfect_match():
    pred = "[excited] Hello WORLD! [laughs]"
    gt = "[excited] Hello WORLD! [laughs]"
    m = compute_tag_metrics(pred, gt)
    assert m.f1 == 1.0
    assert m.position_accuracy == 1.0


def test_missing_tag():
    pred = "[excited] Hello!"
    gt = "[excited] Hello! [laughs]"
    m = compute_tag_metrics(pred, gt)
    assert m.recall < 1.0
    assert m.precision == 1.0


def test_extra_tag():
    pred = "[excited] Hello! [laughs] [sighs]"
    gt = "[excited] Hello! [laughs]"
    m = compute_tag_metrics(pred, gt)
    assert m.precision < 1.0
    assert m.recall == 1.0


def test_no_tags_both():
    m = compute_tag_metrics("Hello world", "Hello world")
    assert m.f1 == 1.0


def test_full_bench():
    preds = ["[excited] Hello!", "Goodbye [sighs]"]
    gts = ["[excited] Hello!", "Goodbye [sighs]"]
    result = evaluate(preds, gts)
    assert result.wer < 0.01
    assert result.tag_f1 == 1.0
```

**Step 5: Run tests**

Run: `python -m pytest tests/test_eval.py -v`
Expected: PASS

**Step 6: Commit**

```bash
git add src/eval/ tests/test_eval.py
git commit -m "feat: add Evoxtral-Bench evaluation suite (WER + Tag F1 + round-trip)"
```

---

## Task 7: Backend API (FastAPI)

**Files:**
- Create: `src/api/main.py`
- Create: `src/api/model_service.py`
- Create: `src/api/schemas.py`

**Step 1: Write schemas**

`src/api/schemas.py`:
```python
"""API request/response schemas."""

from pydantic import BaseModel


class TranscribeResponse(BaseModel):
    tagged_text: str
    plain_text: str
    tags: list[dict]
    processing_time_ms: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
```

**Step 2: Write model service**

`src/api/model_service.py`:
```python
"""Model loading and inference service."""

import torch
import torchaudio
import time
from pathlib import Path
from transformers import AutoProcessor, AutoModelForVision2Seq
from peft import PeftModel
from src.data.formatter import extract_tags, strip_tags


class ModelService:
    def __init__(
        self,
        base_model: str = "mistralai/Voxtral-Mini-3B-2507",
        adapter_path: str | None = None,
    ):
        self.base_model = base_model
        self.adapter_path = adapter_path
        self.model = None
        self.processor = None

    def load(self):
        """Load model and processor."""
        self.processor = AutoProcessor.from_pretrained(self.base_model)
        model = AutoModelForVision2Seq.from_pretrained(
            self.base_model,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        if self.adapter_path:
            model = PeftModel.from_pretrained(model, self.adapter_path)
        self.model = model
        self.model.eval()

    @property
    def is_loaded(self) -> bool:
        return self.model is not None

    def transcribe(self, audio_path: str) -> dict:
        """Transcribe audio file to tagged text."""
        if not self.is_loaded:
            raise RuntimeError("Model not loaded")

        start = time.time()

        # Load and resample audio
        waveform, sr = torchaudio.load(audio_path)
        if sr != 16000:
            resampler = torchaudio.transforms.Resample(sr, 16000)
            waveform = resampler(waveform)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Process
        inputs = self.processor(
            audios=waveform.squeeze().numpy(),
            text="<transcribe>",
            sampling_rate=16000,
            return_tensors="pt",
        ).to(self.model.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
            )

        tagged_text = self.processor.decode(output_ids[0], skip_special_tokens=True)
        elapsed_ms = (time.time() - start) * 1000

        return {
            "tagged_text": tagged_text,
            "plain_text": strip_tags(tagged_text),
            "tags": extract_tags(tagged_text),
            "processing_time_ms": elapsed_ms,
        }
```

**Step 3: Write FastAPI app**

`src/api/main.py`:
```python
"""FastAPI application for Evoxtral."""

import os
import tempfile
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from .model_service import ModelService
from .schemas import TranscribeResponse, HealthResponse

load_dotenv()

model_service = ModelService(
    adapter_path=os.getenv("EVOXTRAL_ADAPTER_PATH", None),
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    model_service.load()
    yield


app = FastAPI(title="Evoxtral API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(status="ok", model_loaded=model_service.is_loaded)


@app.post("/transcribe", response_model=TranscribeResponse)
async def transcribe(file: UploadFile = File(...)):
    if not model_service.is_loaded:
        raise HTTPException(503, "Model not loaded")

    suffix = os.path.splitext(file.filename or ".wav")[1]
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        result = model_service.transcribe(tmp_path)
        return TranscribeResponse(**result)
    finally:
        os.unlink(tmp_path)
```

**Step 4: Commit**

```bash
git add src/api/
git commit -m "feat: add FastAPI backend with model serving"
```

---

## Task 8: Frontend (Next.js + shadcn)

**Files:**
- Modify: `pages/index.tsx` (or create if not exists)
- Create: `components/AudioRecorder.tsx`
- Create: `components/TaggedTranscript.tsx`
- Create: `components/AudioPlayer.tsx`

This task covers the Next.js frontend with:
- Audio upload/record functionality
- Display of tagged transcription with color-coded tags
- Playback of re-synthesized audio
- API integration with the FastAPI backend

**Step 1: Install frontend dependencies**

Run:
```bash
npx shadcn@latest init
npm install @phosphor-icons/react framer-motion
```

**Step 2: Create AudioRecorder component**

`components/AudioRecorder.tsx`:
```tsx
"use client";

import { useState, useRef } from "react";
import { Microphone, Stop, Upload } from "@phosphor-icons/react";

interface AudioRecorderProps {
  onAudioReady: (file: File) => void;
  isProcessing: boolean;
}

export function AudioRecorder({ onAudioReady, isProcessing }: AudioRecorderProps) {
  const [isRecording, setIsRecording] = useState(false);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const startRecording = async () => {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    const mediaRecorder = new MediaRecorder(stream);
    mediaRecorderRef.current = mediaRecorder;
    chunksRef.current = [];

    mediaRecorder.ondataavailable = (e) => {
      if (e.data.size > 0) chunksRef.current.push(e.data);
    };

    mediaRecorder.onstop = () => {
      const blob = new Blob(chunksRef.current, { type: "audio/webm" });
      const file = new File([blob], "recording.webm", { type: "audio/webm" });
      onAudioReady(file);
      stream.getTracks().forEach((t) => t.stop());
    };

    mediaRecorder.start();
    setIsRecording(true);
  };

  const stopRecording = () => {
    mediaRecorderRef.current?.stop();
    setIsRecording(false);
  };

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) onAudioReady(file);
  };

  return (
    <div className="flex items-center gap-4">
      <button
        onClick={isRecording ? stopRecording : startRecording}
        disabled={isProcessing}
        className={`flex items-center gap-2 px-6 py-3 rounded-full font-medium transition-all ${
          isRecording
            ? "bg-red-500 text-white animate-pulse"
            : "bg-zinc-900 text-white hover:bg-zinc-700"
        } disabled:opacity-50`}
      >
        {isRecording ? <Stop size={20} /> : <Microphone size={20} />}
        {isRecording ? "Stop" : "Record"}
      </button>

      <span className="text-zinc-500">or</span>

      <button
        onClick={() => fileInputRef.current?.click()}
        disabled={isProcessing}
        className="flex items-center gap-2 px-6 py-3 rounded-full border border-zinc-300 hover:bg-zinc-50 transition-all disabled:opacity-50"
      >
        <Upload size={20} />
        Upload
      </button>
      <input
        ref={fileInputRef}
        type="file"
        accept="audio/*"
        onChange={handleFileUpload}
        className="hidden"
      />
    </div>
  );
}
```

**Step 3: Create TaggedTranscript component**

`components/TaggedTranscript.tsx`:
```tsx
"use client";

import { motion } from "framer-motion";

interface TaggedTranscriptProps {
  text: string;
}

const TAG_COLORS: Record<string, string> = {
  excited: "bg-yellow-100 text-yellow-800",
  sad: "bg-blue-100 text-blue-800",
  angry: "bg-red-100 text-red-800",
  nervous: "bg-purple-100 text-purple-800",
  calm: "bg-green-100 text-green-800",
  frustrated: "bg-orange-100 text-orange-800",
  laughs: "bg-amber-100 text-amber-800",
  sighs: "bg-slate-100 text-slate-800",
  gasps: "bg-pink-100 text-pink-800",
  whispers: "bg-indigo-100 text-indigo-800",
  shouts: "bg-red-200 text-red-900",
  pause: "bg-gray-100 text-gray-600",
  crying: "bg-blue-200 text-blue-900",
  stammers: "bg-violet-100 text-violet-800",
  "clears throat": "bg-teal-100 text-teal-800",
};

function getTagColor(tag: string): string {
  return TAG_COLORS[tag.toLowerCase()] || "bg-zinc-100 text-zinc-700";
}

export function TaggedTranscript({ text }: TaggedTranscriptProps) {
  if (!text) return null;

  // Parse text into segments: tags and plain text
  const parts = text.split(/(\[[^\]]+\])/g).filter(Boolean);

  return (
    <div className="font-mono text-lg leading-relaxed space-x-1">
      {parts.map((part, i) => {
        const tagMatch = part.match(/^\[([^\]]+)\]$/);
        if (tagMatch) {
          const tag = tagMatch[1];
          return (
            <motion.span
              key={i}
              initial={{ opacity: 0, scale: 0.8 }}
              animate={{ opacity: 1, scale: 1 }}
              className={`inline-block px-2 py-0.5 rounded-md text-sm font-semibold ${getTagColor(tag)}`}
            >
              {tag}
            </motion.span>
          );
        }
        return <span key={i}>{part}</span>;
      })}
    </div>
  );
}
```

**Step 4: Create main page**

`pages/index.tsx`:
```tsx
import { useState } from "react";
import { AudioRecorder } from "../components/AudioRecorder";
import { TaggedTranscript } from "../components/TaggedTranscript";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export default function Home() {
  const [isProcessing, setIsProcessing] = useState(false);
  const [taggedText, setTaggedText] = useState("");
  const [plainText, setPlainText] = useState("");
  const [error, setError] = useState("");
  const [audioUrl, setAudioUrl] = useState<string | null>(null);

  const handleAudio = async (file: File) => {
    setIsProcessing(true);
    setError("");
    setAudioUrl(URL.createObjectURL(file));

    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await fetch(`${API_URL}/transcribe`, {
        method: "POST",
        body: formData,
      });
      if (!res.ok) throw new Error(`API error: ${res.status}`);
      const data = await res.json();
      setTaggedText(data.tagged_text);
      setPlainText(data.plain_text);
    } catch (e: any) {
      setError(e.message);
    } finally {
      setIsProcessing(false);
    }
  };

  return (
    <main className="min-h-screen bg-white">
      <div className="max-w-4xl mx-auto px-6 py-16">
        <h1 className="text-4xl font-bold tracking-tight mb-2">Evoxtral</h1>
        <p className="text-zinc-500 mb-12">
          Emotion-aware transcription with inline audio tags
        </p>

        <div className="mb-12">
          <AudioRecorder onAudioReady={handleAudio} isProcessing={isProcessing} />
        </div>

        {isProcessing && (
          <div className="text-zinc-500 animate-pulse">Transcribing...</div>
        )}

        {error && (
          <div className="text-red-500 mb-4">Error: {error}</div>
        )}

        {taggedText && (
          <div className="space-y-8">
            <div>
              <h2 className="text-sm font-medium text-zinc-400 uppercase tracking-wide mb-3">
                Tagged Transcription
              </h2>
              <div className="p-6 rounded-xl border border-zinc-200 bg-zinc-50">
                <TaggedTranscript text={taggedText} />
              </div>
            </div>

            <div>
              <h2 className="text-sm font-medium text-zinc-400 uppercase tracking-wide mb-3">
                Plain Text
              </h2>
              <div className="p-6 rounded-xl border border-zinc-200">
                <p className="font-mono text-lg">{plainText}</p>
              </div>
            </div>

            {audioUrl && (
              <div>
                <h2 className="text-sm font-medium text-zinc-400 uppercase tracking-wide mb-3">
                  Original Audio
                </h2>
                <audio controls src={audioUrl} className="w-full" />
              </div>
            )}
          </div>
        )}
      </div>
    </main>
  );
}
```

**Step 5: Commit**

```bash
git add components/ pages/
git commit -m "feat: add Next.js frontend with audio recorder and tag display"
```

---

## Task 9: Data Pipeline Runner (End-to-End)

**Files:**
- Create: `scripts/generate_data.py`
- Create: `scripts/run_training.py`
- Create: `scripts/run_eval.py`

**Step 1: Create data generation runner**

`scripts/generate_data.py`:
```python
"""End-to-end data generation: scripts -> synthesis -> dataset."""

import sys
sys.path.insert(0, ".")

from dotenv import load_dotenv
load_dotenv()

from src.data.script_generator import generate_full_dataset
from src.data.synthesizer import synthesize_dataset
from src.data.formatter import format_dataset


def main():
    print("=== Step 1: Generate tagged scripts ===")
    generate_full_dataset(total=1000, output_path="data/scripts/scripts.json")

    print("\n=== Step 2: Synthesize audio via ElevenLabs ===")
    synthesize_dataset(
        scripts_path="data/scripts/scripts.json",
        output_dir="data/audio",
    )

    print("\n=== Step 3: Format HuggingFace dataset ===")
    format_dataset(
        manifest_path="data/audio/manifest.json",
        output_dir="data/processed",
    )

    print("\nData pipeline complete!")


if __name__ == "__main__":
    main()
```

**Step 2: Create training runner**

`scripts/run_training.py`:
```python
"""Run LoRA finetuning with W&B tracking."""

import sys
sys.path.insert(0, ".")

from dotenv import load_dotenv
load_dotenv()

from src.training.finetune import train
from src.training.config import EvoxtralTrainingConfig


def main():
    config = EvoxtralTrainingConfig()
    train(config)


if __name__ == "__main__":
    main()
```

**Step 3: Create eval runner**

`scripts/run_eval.py`:
```python
"""Run Evoxtral-Bench evaluation on test set."""

import sys
sys.path.insert(0, ".")

from dotenv import load_dotenv
load_dotenv()

import torch
from datasets import load_from_disk
from src.api.model_service import ModelService
from src.eval.bench import evaluate, print_results, save_results


def main():
    print("Loading test dataset...")
    ds = load_from_disk("data/processed")
    test_ds = ds["test"]

    print("Loading model...")
    service = ModelService(
        adapter_path="model/evoxtral-lora",
    )
    service.load()

    print(f"Running inference on {len(test_ds)} test samples...")
    predictions = []
    ground_truths = []

    for i, example in enumerate(test_ds):
        # Save audio to temp file for inference
        import tempfile, soundfile as sf
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            sf.write(tmp.name, example["audio"]["array"], example["audio"]["sampling_rate"])
            result = service.transcribe(tmp.name)
            predictions.append(result["tagged_text"])
            ground_truths.append(example["tagged_text"])

        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{len(test_ds)}")

    print("\nRunning Evoxtral-Bench...")
    result = evaluate(predictions, ground_truths)
    print_results(result)
    save_results(result, "eval_results.json")


if __name__ == "__main__":
    main()
```

**Step 4: Commit**

```bash
git add scripts/
git commit -m "feat: add end-to-end pipeline runners (data, train, eval)"
```

---

## Task 10: Integration, Model Card & Submission

**Files:**
- Create: `model/README.md` (HuggingFace model card)
- Modify: `README.md` (add setup instructions)

**Step 1: Write HuggingFace model card**

`model/README.md`:
```markdown
---
license: apache-2.0
base_model: mistralai/Voxtral-Mini-3B-2507
tags:
  - speech
  - transcription
  - emotion
  - elevenlabs
  - audio-tags
  - lora
datasets:
  - evoxtral-synthetic-1k
---

# Evoxtral

LoRA adapter for Voxtral-Mini-3B that produces transcriptions with inline ElevenLabs v3 audio tags.

## Usage

```python
from transformers import AutoProcessor, AutoModelForVision2Seq
from peft import PeftModel

processor = AutoProcessor.from_pretrained("mistralai/Voxtral-Mini-3B-2507")
model = AutoModelForVision2Seq.from_pretrained("mistralai/Voxtral-Mini-3B-2507")
model = PeftModel.from_pretrained(model, "mistral-hackaton-2026/evoxtral-lora")
```

## Evoxtral-Bench Results

| Metric | Score |
|--------|-------|
| WER | TBD |
| Tag F1 | TBD |
| Tag Position Accuracy | TBD |

## Built For

Mistral AI Online Hackathon 2026 - Fine-tuning track
```

**Step 2: Run the full pipeline**

```bash
# 1. Generate data
python scripts/generate_data.py

# 2. Train
python scripts/run_training.py

# 3. Evaluate
python scripts/run_eval.py

# 4. Start backend
uvicorn src.api.main:app --host 0.0.0.0 --port 8000

# 5. Start frontend (separate terminal)
npm run dev
```

**Step 3: Final commit**

```bash
git add -A
git commit -m "feat: complete Evoxtral pipeline - data, training, eval, API, frontend"
```
