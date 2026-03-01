"""Generate diverse tagged scripts for ElevenLabs v3 TTS synthesis.

Uses async concurrency for fast generation (~2 min for 1000 scripts).
"""

import asyncio
import json
import random
import os
import sys
from pathlib import Path
from anthropic import AsyncAnthropic
from dotenv import load_dotenv
from .tag_taxonomy import ALL_BRACKET_TAGS, SLICE_CONFIG, DOMAINS

load_dotenv()

SYSTEM_PROMPT = """You are a script writer generating realistic speech samples with inline ElevenLabs v3 audio tags.

Audio tags use square brackets: [laughs], [excited], [whispers], [pause], etc.
Emphasis uses CAPITALIZATION of stressed words.
Pauses use ellipses ...

Rules:
- Write natural, diverse dialogue/monologue snippets (15-80 words each)
- Tags must feel organic, not forced
- Vary domains: conversation, podcast, storytelling, argument, etc.
- Include a mix of male/female perspectives and speaking styles
- Each sample should be self-contained (no context needed)
- NEVER use tags that aren't in the available list
- For CAPS emphasis, only capitalize 1-3 key words per sentence, not whole sentences

Available tags (use ONLY these in brackets): {tags}

Output ONLY a valid JSON array of objects. Each object has these fields:
- "tagged_text": the text with inline tags (string)
- "plain_text": same text with all [tags] removed and CAPS converted to lowercase (string)
- "tags_used": list of tag names used without brackets (array of strings)
- "domain": one of {domains} (string)
- "tag_count": number of bracket tags used (integer)

Example output:
[
  {{
    "tagged_text": "[excited] I can't BELIEVE we actually won! [laughs] This is incredible...",
    "plain_text": "I can't believe we actually won! This is incredible...",
    "tags_used": ["excited", "laughs"],
    "domain": "conversation",
    "tag_count": 2
  }}
]"""


async def generate_batch(
    client: AsyncAnthropic,
    slice_name: str,
    batch_num: int,
    count: int,
    config: dict,
    model: str,
) -> list[dict]:
    """Generate a single batch of scripts."""
    tag_density = config["tag_density"]

    if slice_name == "plain":
        density_instruction = "Do NOT include any audio tags [brackets] or CAPS emphasis. Write completely plain, natural speech only. No tags at all."
    elif isinstance(tag_density, tuple):
        density_instruction = f"Use exactly {tag_density[0]} to {tag_density[1]} audio tags per sample."
    else:
        density_instruction = f"Use exactly {tag_density} audio tags per sample."

    if slice_name == "edge":
        density_instruction += (
            " Include edge cases: tags at the very start and end, "
            "consecutive tags like [angry] [laughs], "
            "ambiguous emotions, very short utterances with tags, "
            "and long utterances with scattered tags."
        )

    domain_subset = random.sample(DOMAINS, min(4, len(DOMAINS)))
    prompt = (
        f"Generate exactly {count} unique speech samples.\n"
        f"Slice type: {slice_name} — {config['description']}\n"
        f"Tag density: {density_instruction}\n"
        f"Distribute across these domains: {', '.join(domain_subset)}\n"
        f"Make each sample UNIQUE — different topics, tones, speakers, situations.\n"
        f"Vary sentence length: mix short (15-25 words), medium (25-45 words), and long (45-80 words)."
    )

    try:
        response = await client.messages.create(
            model=model,
            max_tokens=4096,
            system=SYSTEM_PROMPT.format(
                tags=", ".join(ALL_BRACKET_TAGS),
                domains=", ".join(DOMAINS),
            ),
            messages=[{"role": "user", "content": prompt}],
        )

        content = response.content[0].text
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            content = content.split("```")[1].split("```")[0]

        batch = json.loads(content.strip())
        if not isinstance(batch, list):
            batch = batch.get("samples", batch.get("scripts", [batch]))

        for item in batch:
            item["slice_type"] = slice_name

        print(f"  [{slice_name}] Batch {batch_num + 1}: {len(batch)} scripts", flush=True)
        return batch

    except Exception as e:
        print(f"  [{slice_name}] Batch {batch_num + 1} FAILED: {e}", flush=True)
        return []


async def generate_slice(
    client: AsyncAnthropic,
    slice_name: str,
    count: int,
    model: str,
    max_concurrent: int = 10,
) -> list[dict]:
    """Generate all scripts for a slice with concurrent batches."""
    config = SLICE_CONFIG[slice_name]
    batch_size = 25
    num_batches = (count + batch_size - 1) // batch_size

    semaphore = asyncio.Semaphore(max_concurrent)

    async def limited_batch(batch_num: int, n: int):
        async with semaphore:
            return await generate_batch(client, slice_name, batch_num, n, config, model)

    tasks = []
    for i in range(num_batches):
        n = min(batch_size, count - i * batch_size)
        tasks.append(limited_batch(i, n))

    results = await asyncio.gather(*tasks)
    scripts = [s for batch in results for s in batch]
    return scripts[:count]


async def generate_full_dataset_async(
    total: int = 1000,
    output_path: str = "data/scripts/scripts.json",
    model: str = "claude-sonnet-4-5-20250929",
    max_concurrent: int = 10,
) -> list[dict]:
    """Generate the full balanced dataset with async concurrency."""
    client = AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    all_scripts = []

    for slice_name, config in SLICE_CONFIG.items():
        count = int(total * config["ratio"])
        print(f"\nGenerating {count} '{slice_name}' scripts ({(count + 24) // 25} batches, {max_concurrent} concurrent)...", flush=True)
        scripts = await generate_slice(client, slice_name, count, model, max_concurrent)
        all_scripts.extend(scripts)
        print(f"  Total for {slice_name}: {len(scripts)}", flush=True)

    random.shuffle(all_scripts)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_scripts, f, indent=2)

    print(f"\n{'=' * 50}", flush=True)
    print(f"DATASET GENERATION COMPLETE", flush=True)
    print(f"{'=' * 50}", flush=True)
    print(f"Total scripts: {len(all_scripts)}", flush=True)
    for sn in SLICE_CONFIG:
        c = sum(1 for s in all_scripts if s.get("slice_type") == sn)
        print(f"  {sn}: {c}", flush=True)
    print(f"Saved to: {output_path}", flush=True)

    return all_scripts


def generate_full_dataset(total: int = 1000, output_path: str = "data/scripts/scripts.json") -> list[dict]:
    """Sync wrapper for async generation."""
    return asyncio.run(generate_full_dataset_async(total, output_path))


if __name__ == "__main__":
    generate_full_dataset()
