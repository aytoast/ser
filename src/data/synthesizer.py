"""ElevenLabs v3 TTS synthesizer with async concurrency, key rotation, and crash resilience.

Features:
- Async parallel synthesis (up to N concurrent per key)
- Automatic key rotation across multiple API keys
- Incremental saves: manifest updated after every batch (crash-safe)
- Resume support: skips already-synthesized files on restart
- Voice diversity: round-robin across 8 voices
"""

import asyncio
import json
import hashlib
import os
import time
from pathlib import Path
from collections import Counter
from dataclasses import dataclass, asdict
from dotenv import load_dotenv

load_dotenv()

MODEL_ID = "eleven_v3"

VOICE_POOL = [
    {"id": "CwhRBWXzGAHq8TQ4Fs17", "name": "Roger", "gender": "male", "age": "middle_aged"},
    {"id": "cjVigY5qzO86Huf0OWal", "name": "Eric", "gender": "male", "age": "middle_aged"},
    {"id": "EXAVITQu4vr4xnSDxMaL", "name": "Sarah", "gender": "female", "age": "young"},
    {"id": "XrExE9yKIg1WjnnlVkGX", "name": "Matilda", "gender": "female", "age": "middle_aged"},
    {"id": "TX3LPaxmHKxFdv7VOQHJ", "name": "Liam", "gender": "male", "age": "young"},
    {"id": "cgSgspJ2msm6clMCkdW9", "name": "Jessica", "gender": "female", "age": "young"},
    {"id": "JBFqnCBsd6RMkjVDRZzb", "name": "George", "gender": "male", "age": "middle_aged"},
    {"id": "pFZP5JQG7iQjIQuC4Bku", "name": "Lily", "gender": "female", "age": "middle_aged"},
]


@dataclass
class SynthesisResult:
    script_index: int
    audio_path: str
    voice_id: str
    voice_name: str
    success: bool
    duration_ms: float = 0.0
    error: str | None = None
    skipped: bool = False


def get_api_keys() -> list[str]:
    """Load all working ElevenLabs API keys from environment."""
    keys = []
    for key_name, value in sorted(os.environ.items()):
        if key_name.startswith("ELEVENLABS_API_KEY"):
            keys.append(value)
    if not keys:
        raise ValueError("No ELEVENLABS_API_KEY* found in environment")
    return keys


def load_progress(manifest_path: str) -> dict[int, dict]:
    """Load existing manifest to resume from where we left off."""
    if not Path(manifest_path).exists():
        return {}
    with open(manifest_path) as f:
        manifest = json.load(f)
    return {item["_index"]: item for item in manifest if item.get("synthesis_success")}


def save_manifest(manifest: list[dict], manifest_path: str):
    """Save manifest incrementally (atomic write)."""
    tmp_path = manifest_path + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(manifest, f, indent=2)
    os.replace(tmp_path, manifest_path)


async def synthesize_one(
    api_key: str,
    text: str,
    voice_id: str,
    output_path: str,
    semaphore: asyncio.Semaphore,
) -> bool:
    """Synthesize a single text to audio file."""
    import httpx

    async with semaphore:
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}",
                    headers={
                        "xi-api-key": api_key,
                        "Content-Type": "application/json",
                    },
                    json={
                        "text": text,
                        "model_id": MODEL_ID,
                        "output_format": "mp3_44100_128",
                    },
                )

                if response.status_code == 200:
                    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
                    with open(output_path, "wb") as f:
                        f.write(response.content)
                    return True
                else:
                    error_msg = response.text[:100]
                    # Don't print rate limit errors for every hit
                    if response.status_code != 429:
                        print(f"    HTTP {response.status_code}: {error_msg}", flush=True)
                    return False
        except Exception as e:
            print(f"    Network error: {e}", flush=True)
            return False


async def synthesize_dataset(
    scripts_path: str = "data/scripts/scripts.json",
    output_dir: str = "data/audio",
    max_concurrent_per_key: int = 5,
    max_retries: int = 2,
    batch_size: int = 20,
) -> list[SynthesisResult]:
    """Synthesize all scripts with crash resilience and incremental saves."""
    with open(scripts_path) as f:
        scripts = json.load(f)

    keys = get_api_keys()
    print(f"API keys loaded: {len(keys)}", flush=True)

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    manifest_path = f"{output_dir}/manifest.json"

    # Load progress from previous run
    completed = load_progress(manifest_path)
    if completed:
        print(f"Resuming: {len(completed)} already done, {len(scripts) - len(completed)} remaining", flush=True)

    # Build manifest (preserving completed entries)
    manifest = []
    for i, script in enumerate(scripts):
        if i in completed:
            manifest.append(completed[i])
        else:
            manifest.append({
                **script,
                "_index": i,
                "audio_path": None,
                "voice_id": None,
                "voice_name": None,
                "synthesis_success": False,
            })

    # Create semaphores per key
    key_semaphores = [(key, asyncio.Semaphore(max_concurrent_per_key)) for key in keys]
    key_idx = [0]  # mutable counter for round-robin

    def next_key_and_sem():
        idx = key_idx[0] % len(key_semaphores)
        key_idx[0] += 1
        return key_semaphores[idx]

    # Voice assignment: deterministic per index
    def get_voice(i: int):
        return VOICE_POOL[i % len(VOICE_POOL)]

    async def process_one(i: int, script: dict) -> SynthesisResult:
        voice = get_voice(i)
        content_hash = hashlib.md5(script["tagged_text"].encode()).hexdigest()[:8]
        audio_path = f"{output_dir}/{i:04d}_{content_hash}.mp3"

        # Skip if already done
        if Path(audio_path).exists() and Path(audio_path).stat().st_size > 500:
            return SynthesisResult(i, audio_path, voice["id"], voice["name"], True, skipped=True)

        start = time.time()
        success = False
        error = None

        for attempt in range(max_retries + 1):
            api_key, sem = next_key_and_sem()
            success = await synthesize_one(api_key, script["tagged_text"], voice["id"], audio_path, sem)
            if success:
                break
            error = f"Attempt {attempt + 1} failed"
            if attempt < max_retries:
                await asyncio.sleep(1.0 * (attempt + 1))

        elapsed_ms = (time.time() - start) * 1000
        return SynthesisResult(i, audio_path, voice["id"], voice["name"], success, elapsed_ms, error)

    # Process in batches with incremental saves
    all_results: list[SynthesisResult] = []
    pending_indices = [i for i in range(len(scripts)) if i not in completed]

    total_ok = len(completed)
    total_skip = len(completed)
    total_fail = 0

    for batch_start in range(0, len(pending_indices), batch_size):
        batch_indices = pending_indices[batch_start:batch_start + batch_size]
        batch_tasks = [process_one(i, scripts[i]) for i in batch_indices]
        batch_results = await asyncio.gather(*batch_tasks)

        # Update manifest with results
        batch_ok = 0
        for r in batch_results:
            manifest[r.script_index] = {
                **scripts[r.script_index],
                "_index": r.script_index,
                "audio_path": r.audio_path,
                "voice_id": r.voice_id,
                "voice_name": r.voice_name,
                "synthesis_success": r.success,
                "synthesis_duration_ms": r.duration_ms,
            }
            if r.success:
                batch_ok += 1
                total_ok += 1
            else:
                total_fail += 1
            all_results.append(r)

        # Incremental save after every batch
        save_manifest(manifest, manifest_path)

        batch_num = batch_start // batch_size + 1
        total_batches = (len(pending_indices) + batch_size - 1) // batch_size
        print(
            f"  Batch {batch_num}/{total_batches}: "
            f"{batch_ok}/{len(batch_results)} ok | "
            f"Total: {total_ok}/{len(scripts)} done, {total_fail} failed | "
            f"Saved to manifest",
            flush=True,
        )

    # Final summary
    print(f"\n{'=' * 50}", flush=True)
    print(f"SYNTHESIS COMPLETE", flush=True)
    print(f"{'=' * 50}", flush=True)
    print(f"Success:  {total_ok}/{len(scripts)}", flush=True)
    print(f"Skipped:  {total_skip} (already existed)", flush=True)
    print(f"Failed:   {total_fail}", flush=True)
    print(f"Manifest: {manifest_path}", flush=True)

    voice_dist = Counter()
    for m in manifest:
        if m.get("synthesis_success"):
            voice_dist[m.get("voice_name", "?")] += 1
    print(f"\nVoice distribution:", flush=True)
    for name, count in voice_dist.most_common():
        pct = count / max(total_ok, 1) * 100
        print(f"  {name}: {count} ({pct:.1f}%)", flush=True)

    return all_results


def run():
    """Entry point."""
    asyncio.run(synthesize_dataset())


if __name__ == "__main__":
    run()
