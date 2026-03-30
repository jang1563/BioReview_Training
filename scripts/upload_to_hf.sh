#!/usr/bin/env bash
# =============================================================================
# upload_to_hf.sh — Upload Qwen3.5-9B SFT adapter to Hugging Face Hub (private)
#
# Run this from the project root on HPC (Cornell Cayuga):
#   bash scripts/upload_to_hf.sh
#
# Requirements:
#   pip install huggingface_hub
#   export HF_TOKEN=hf_...   (or run: huggingface-cli login)
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
HF_REPO="jang1563/bioreview-qwen3.5-9b-sft"
MODEL_DIR="${MODEL_DIR:-models/qwen3.5_9b_all_nonfig_v1}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# ---------------------------------------------------------------------------
# Validate environment
# ---------------------------------------------------------------------------
echo "=== BioReview HF Upload ==="
echo "Repo   : ${HF_REPO}"
echo "Source : ${PROJECT_ROOT}/${MODEL_DIR}"
echo ""

if [[ -z "${HF_TOKEN:-}" ]]; then
    echo "[INFO] HF_TOKEN not set — falling back to cached credentials."
    echo "       If this fails, run: huggingface-cli login"
fi

if ! python -c "import huggingface_hub" 2>/dev/null; then
    echo "[ERROR] huggingface_hub not installed. Run: pip install huggingface_hub"
    exit 1
fi

ADAPTER_DIR="${PROJECT_ROOT}/${MODEL_DIR}"
if [[ ! -d "${ADAPTER_DIR}" ]]; then
    echo "[ERROR] Model directory not found: ${ADAPTER_DIR}"
    echo "        Set MODEL_DIR env var to the correct checkpoint path."
    exit 1
fi

# ---------------------------------------------------------------------------
# Create private repo (idempotent — safe to re-run)
# ---------------------------------------------------------------------------
echo "[1/3] Creating private HF repo (if it doesn't exist)..."
python - <<EOF
from huggingface_hub import HfApi
import os

api = HfApi(token=os.environ.get("HF_TOKEN"))
try:
    api.create_repo(
        repo_id="${HF_REPO}",
        repo_type="model",
        private=True,
        exist_ok=True,
    )
    print("      Repo ready: https://huggingface.co/${HF_REPO}")
except Exception as e:
    print(f"[ERROR] {e}")
    raise
EOF

# ---------------------------------------------------------------------------
# Upload LoRA adapter files
# ---------------------------------------------------------------------------
echo "[2/3] Uploading adapter weights from: ${ADAPTER_DIR}"
python - <<EOF
from huggingface_hub import HfApi
import os, pathlib

api   = HfApi(token=os.environ.get("HF_TOKEN"))
src   = pathlib.Path("${ADAPTER_DIR}")

# Patterns to include (LoRA adapter files only — NOT base model weights)
INCLUDE = {
    "adapter_config.json",
    "adapter_model.safetensors",
    "adapter_model.bin",          # fallback if no safetensors
    "tokenizer_config.json",
    "tokenizer.json",
    "tokenizer.model",
    "special_tokens_map.json",
    "generation_config.json",
    "config.json",
}

uploaded = []
skipped  = []

for f in sorted(src.rglob("*")):
    if not f.is_file():
        continue
    # Skip optimizer states, full checkpoints, and training artifacts
    if any(skip in f.name for skip in [
        "optimizer", "scheduler", "rng_state",
        "training_args", "trainer_state",
    ]):
        skipped.append(f.name)
        continue
    if f.name not in INCLUDE and not f.name.endswith(".safetensors"):
        skipped.append(f.name)
        continue

    rel = f.relative_to(src)
    print(f"  Uploading: {rel}  ({f.stat().st_size / 1e6:.1f} MB)")
    api.upload_file(
        path_or_fileobj=str(f),
        path_in_repo=str(rel),
        repo_id="${HF_REPO}",
        repo_type="model",
        token=os.environ.get("HF_TOKEN"),
    )
    uploaded.append(str(rel))

print(f"\n  Uploaded {len(uploaded)} file(s). Skipped {len(skipped)} file(s).")
if skipped:
    print(f"  Skipped: {', '.join(skipped[:5])}{'...' if len(skipped) > 5 else ''}")
EOF

# ---------------------------------------------------------------------------
# Upload model card and training config
# ---------------------------------------------------------------------------
echo "[3/3] Uploading model card and training config..."
python - <<EOF
from huggingface_hub import HfApi
import os

api   = HfApi(token=os.environ.get("HF_TOKEN"))
root  = "${PROJECT_ROOT}"

uploads = [
    ("MODEL_CARD.md",                              "README.md"),
    ("configs/qwen3.5_9b_all_nonfig.yaml",         "training_config.yaml"),
    ("requirements-inference.txt",                 "requirements.txt"),
]

for local_rel, repo_path in uploads:
    local = os.path.join(root, local_rel)
    if not os.path.exists(local):
        print(f"  [SKIP] Not found: {local_rel}")
        continue
    print(f"  Uploading: {local_rel} -> {repo_path}")
    api.upload_file(
        path_or_fileobj=local,
        path_in_repo=repo_path,
        repo_id="${HF_REPO}",
        repo_type="model",
        token=os.environ.get("HF_TOKEN"),
    )

print("  Done.")
EOF

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo ""
echo "=== Upload complete ==="
echo "Private model repo: https://huggingface.co/${HF_REPO}"
echo ""
echo "To make public later, run:"
echo ""
cat <<'MAKE_PUBLIC'
  python - <<EOF
  from huggingface_hub import HfApi
  HfApi().update_repo_visibility("jang1563/bioreview-qwen3.5-9b-sft", private=False)
  print("Repo is now public.")
  EOF
MAKE_PUBLIC
