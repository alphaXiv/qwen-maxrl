#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export ARTIFACTS_DIR="${ARTIFACTS_DIR:-$SCRIPT_DIR/.openresearch/artifacts}"

echo "uv version: $(uv --version)"
export UV_TORCH_BACKEND=auto

echo "===== GPU DIAGNOSTICS ====="
echo "--- nvidia-smi ---"
nvidia-smi || echo "nvidia-smi FAILED (no driver or no GPU attached)"
echo "--- nvidia-smi -L ---"
nvidia-smi -L || true
echo "--- CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>} ---"
echo "--- NVIDIA_VISIBLE_DEVICES=${NVIDIA_VISIBLE_DEVICES:-<unset>} ---"
echo "--- /dev/nvidia* ---"
ls -la /dev/nvidia* 2>&1 || true
echo "--- nvidia driver version ---"
cat /proc/driver/nvidia/version 2>&1 || true
echo "--- torch view of CUDA ---"
uv run --python 3.11 --with torch --no-project python - <<'PY' || true
import torch
print("torch:", torch.__version__)
print("torch.version.cuda:", torch.version.cuda)
print("is_available:", torch.cuda.is_available())
print("device_count:", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    try:
        print(f"  [{i}] name:", torch.cuda.get_device_name(i))
        free, total = torch.cuda.mem_get_info(i)
        print(f"  [{i}] mem free/total: {free/1e9:.2f}G / {total/1e9:.2f}G")
    except Exception as e:
        print(f"  [{i}] ERROR:", type(e).__name__, e)
PY
echo "===== END GPU DIAGNOSTICS ====="

GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n1 || echo unknown)"
case "$GPU_NAME" in
  *H100*|*H200*) ARCH=hopper ;;
  *A100*)        ARCH=ampere ;;
  *B200*)        ARCH=blackwell ;;
  *)             ARCH=unknown ;;
esac
echo "Detected GPU: '$GPU_NAME' -> $ARCH"

if [ "$ARCH" = "hopper" ]; then
  export UNSLOTH_VLLM_STANDBY=0
  export GPU_MEM_UTIL=0.7
elif [ "$ARCH" = "blackwell" ]; then
  export UNSLOTH_VLLM_STANDBY=0
  export GPU_MEM_UTIL=0.7
  # flashinfer JIT compilation fails on sm_100a with older CUDA toolkits;
  # fall back to vllm's native torch sampler
  export VLLM_USE_FLASHINFER_SAMPLER=0
else
  export UNSLOTH_VLLM_STANDBY=1
  export GPU_MEM_UTIL=0.9
fi
echo "UNSLOTH_VLLM_STANDBY=$UNSLOTH_VLLM_STANDBY GPU_MEM_UTIL=$GPU_MEM_UTIL VLLM_USE_FLASHINFER_SAMPLER=${VLLM_USE_FLASHINFER_SAMPLER:-<unset>}"

# Only report to wandb if an API key is available; otherwise disable logging.
if [ -n "${WANDB_API_KEY:-}" ]; then
  export REPORT_TO=wandb
else
  export REPORT_TO=none
  echo "WANDB_API_KEY not set -> REPORT_TO=none (wandb logging disabled)"
fi

exec uv run --python 3.11 "$SCRIPT_DIR/qwen-maxrl.py" "$@"
