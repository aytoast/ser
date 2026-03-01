"""Push the trained LoRA adapter from Modal volume to HF Hub."""
import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("huggingface_hub")
)

app = modal.App("evoxtral-push-hub", image=image)
output_vol = modal.Volume.from_name("evoxtral-output", create_if_missing=True)

@app.function(
    volumes={"/output": output_vol},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=600,
)
def push(repo_id: str = "YongkangZOU/evoxtral-lora"):
    import os
    from pathlib import Path
    from huggingface_hub import HfApi

    adapter_dir = "/output/evoxtral-lora"
    if not Path(adapter_dir).exists():
        print(f"ERROR: {adapter_dir} not found on volume")
        return

    files = list(Path(adapter_dir).rglob("*"))
    print(f"Found {len(files)} files in {adapter_dir}")
    for f in files[:20]:
        print(f"  {f}")

    api = HfApi(token=os.environ.get("HF_TOKEN"))
    api.create_repo(repo_id, repo_type="model", exist_ok=True)
    api.upload_folder(
        folder_path=adapter_dir,
        repo_id=repo_id,
        repo_type="model",
        commit_message="LoRA adapter: Voxtral-Mini-3B finetuned for expressive tagged transcription",
    )
    print(f"Successfully pushed to https://huggingface.co/{repo_id}")

    # Also try pushing to org namespace
    org_repo = "mistral-hackaton-2026/evoxtral-lora"
    try:
        api.create_repo(org_repo, repo_type="model", exist_ok=True)
        api.upload_folder(
            folder_path=adapter_dir,
            repo_id=org_repo,
            repo_type="model",
            commit_message="LoRA adapter: Voxtral-Mini-3B finetuned for expressive tagged transcription",
        )
        print(f"Also pushed to https://huggingface.co/{org_repo}")
    except Exception as e:
        print(f"Org push failed (expected): {e}")
        print(f"You can fork/copy from https://huggingface.co/{repo_id} to the org manually.")


@app.local_entrypoint()
def main():
    push.remote()
