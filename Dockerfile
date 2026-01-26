# Debian/Ubuntu-based images, avoid Alpine (musl) for onnxruntime needed by flashrank
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000 \
    FLASHRANK_CACHE_DIR=/opt/flashrank_models \
    FLASHRANK_MAX_LENGTH=128 \
    FLASHRANK_WARMUP_RETRIES=8 \
    FLASHRANK_WARMUP_RETRY_SLEEP=5 \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH="/opt/venv/bin:$PATH" \
    UV_EXTRA_INDEX_URL="https://abetlen.github.io/llama-cpp-python/whl/cpu https://download.pytorch.org/whl/cpu" \
    UV_INDEX_STRATEGY="unsafe-best-match" \
    HF_HOME=/opt/hf \
    HF_HUB_CACHE=/opt/hf/hub \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

WORKDIR /app

# OS deps + toolchain (listwise may require builds)
RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates \
      curl \
      git \
      build-essential \
      gcc \
      g++ \
      make \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh \
 && mv /root/.local/bin/uv /usr/local/bin/uv

# ---------- deps layer ----------
COPY pyproject.toml /app/pyproject.toml
COPY uv.lock /app/uv.lock

RUN uv venv /opt/venv \
 && uv sync --frozen --no-dev

# Optional sanity check (cheap)
RUN python -c "import torch; print('torch:', torch.__version__); import transformers; print('transformers:', transformers.__version__)"

# ---- Bake HF model into image for airgapped deployment ----
ARG HF_MODEL_ID="facebook/bart-large-mnli"
ENV HF_MODEL_ID="${HF_MODEL_ID}"

RUN mkdir -p "$HF_HOME" "$HF_HUB_CACHE" && \
cat > /tmp/bake_hf_model.py <<'PY'
import os
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_id = os.environ.get("HF_MODEL_ID", "facebook/bart-large-mnli")
cache_dir = os.environ.get("HF_HUB_CACHE", "/opt/hf/hub")

snapshot_dir = snapshot_download(repo_id=model_id, cache_dir=cache_dir)
AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir)
AutoModelForSequenceClassification.from_pretrained(model_id, cache_dir=cache_dir)

print("Baked:", model_id)
print("Snapshot:", snapshot_dir)
PY

RUN HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 python /tmp/bake_hf_model.py && \
    rm -f /tmp/bake_hf_model.py

# ---------- bake/warm flashrank models (cached unless MODELS changes) ----------
ARG MODELS="ms-marco-TinyBERT-L-2-v2,ms-marco-MiniLM-L-12-v2,rank-T5-flan,ms-marco-MultiBERT-L-12,rank_zephyr_7b_v1_full"
ENV MODELS="${MODELS}"

RUN set -eux; \
    echo "FLASHRANK_CACHE_DIR=${FLASHRANK_CACHE_DIR}"; \
    echo "FLASHRANK_MAX_LENGTH=${FLASHRANK_MAX_LENGTH}"; \
    echo "MODELS=${MODELS}"; \
    echo "FLASHRANK_WARMUP_RETRIES=${FLASHRANK_WARMUP_RETRIES}"; \
    echo "FLASHRANK_WARMUP_RETRY_SLEEP=${FLASHRANK_WARMUP_RETRY_SLEEP}"; \
    mkdir -p "${FLASHRANK_CACHE_DIR}"; \
    python -u - <<'PY'
import os, time, traceback, subprocess, zipfile, re, shutil
from flashrank import Ranker, RerankRequest

HF_BASE = "https://huggingface.co/prithivida/flashrank/resolve/main"

def curl_head_content_length(url: str) -> int:
    cmd = ["curl", "-sSIL", "--fail", "--http1.1", url]
    out = subprocess.check_output(cmd, text=True)
    lengths = re.findall(r"(?im)^content-length:\s*(\d+)\s*$", out)
    return int(lengths[-1]) if lengths else 0

def resumable_download_until_complete(url: str, dest_path: str, expected_size: int) -> None:
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    while True:
        current = os.path.getsize(dest_path) if os.path.exists(dest_path) else 0
        if expected_size and current >= expected_size:
            print(f"Download complete (size={current}, expected={expected_size})", flush=True)
            return

        print(
            f"Downloading/resuming: {url} -> {dest_path} "
            f"(current={current}, expected={expected_size or 'unknown'})",
            flush=True,
        )
        cmd = [
            "curl", "-L",
            "--fail", "--show-error",
            "--http1.1",
            "--retry", "50",
            "--retry-all-errors",
            "--retry-delay", "5",
            "--connect-timeout", "30",
            "--max-time", "0",
            "-C", "-",              # resume
            "-o", dest_path,
            url,
        ]
        print("Running:", " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True)

def _dir_has_model_files(path: str) -> bool:
    if not os.path.isdir(path):
        return False
    for root, _, files in os.walk(path):
        for f in files:
            if f.endswith(".onnx") or f.endswith(".gguf"):
                return True
        break
    return False

def extract_zip_flatten(zip_path: str, cache_dir: str, model_name: str) -> None:
    """
    Extract zip into a temp dir, then ensure final layout is:
      <cache_dir>/<model_name>/<model files...>
    Handles zips that contain a top-level folder by flattening if needed.
    """
    target_dir = os.path.join(cache_dir, model_name)
    tmp_dir = os.path.join(cache_dir, f".__tmp_extract_{model_name}")
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir, ignore_errors=True)
    os.makedirs(tmp_dir, exist_ok=True)

    print(f"Extracting {zip_path} -> {tmp_dir}", flush=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(tmp_dir)

    entries = [e for e in os.listdir(tmp_dir) if e and e != "__MACOSX"]
    if len(entries) == 1 and os.path.isdir(os.path.join(tmp_dir, entries[0])):
        extracted_root = os.path.join(tmp_dir, entries[0])
    else:
        extracted_root = tmp_dir

    if os.path.exists(target_dir):
        shutil.rmtree(target_dir, ignore_errors=True)
    os.makedirs(os.path.dirname(target_dir), exist_ok=True)

    print(f"Placing model files into {target_dir}", flush=True)
    shutil.move(extracted_root, target_dir)

    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir, ignore_errors=True)

def ensure_model_present(cache_dir: str, model_name: str) -> str:
    """
    Ensure <cache_dir>/<model_name>/ is populated with model files by:
    - resumable downloading <model_name>.zip
    - extracting and flattening into the expected directory
    Returns the model_dir path.
    """
    model_dir = os.path.join(cache_dir, model_name)
    if _dir_has_model_files(model_dir):
        print(f"Model already extracted: {model_dir}", flush=True)
        return model_dir

    url = f"{HF_BASE}/{model_name}.zip"
    zip_path = os.path.join(cache_dir, f"{model_name}.zip")

    expected = 0
    try:
        expected = curl_head_content_length(url)
        print(f"Expected size for {model_name}.zip: {expected or 'unknown'} bytes", flush=True)
    except Exception as e:
        print(f"HEAD size check failed for {model_name}: {e} (continuing anyway)", flush=True)

    resumable_download_until_complete(url, zip_path, expected)
    extract_zip_flatten(zip_path, cache_dir, model_name)

    # Remove zip to save space (optional; comment out if you prefer keeping it)
    try:
        os.remove(zip_path)
    except OSError:
        pass

    return model_dir

def is_listwise_model_dir(model_dir: str) -> bool:
    # If it contains a .gguf, treat it as listwise/llama.cpp model (avoid loading during build)
    for root, _, files in os.walk(model_dir):
        for f in files:
            if f.endswith(".gguf"):
                return True
        break
    return False

try:
    cache_dir = os.environ.get("FLASHRANK_CACHE_DIR", "/opt/flashrank_models")
    max_length = int(os.environ.get("FLASHRANK_MAX_LENGTH", "128"))

    models_env = os.getenv("MODELS", "").strip()
    print("MODELS env raw:", repr(models_env), flush=True)
    if not models_env:
        raise SystemExit("MODELS env var is empty. Set it via Dockerfile ENV or --build-arg MODELS=...")

    models = [m.strip() for m in models_env.split(",") if m.strip()]
    print("Warming up models:", models, flush=True)
    print("Cache dir:", cache_dir, "max_length:", max_length, flush=True)

    retries = int(os.getenv("FLASHRANK_WARMUP_RETRIES", "5"))
    sleep_sec = float(os.getenv("FLASHRANK_WARMUP_RETRY_SLEEP", "3"))

    req = RerankRequest(
        query="warmup",
        passages=[{"id": 1, "text": "warmup one"}, {"id": 2, "text": "warmup two"}],
    )

    for name in models:
        last_exc = None
        for attempt in range(1, retries + 1):
            try:
                print(f"Preparing: {name} (attempt {attempt}/{retries})", flush=True)

                # Always download+extract ourselves (resumable) to bake into the image
                model_dir = ensure_model_present(cache_dir, name)

                # IMPORTANT: don't load listwise LLM during build (can OOM / kill the build)
                if is_listwise_model_dir(model_dir):
                    print(f"Skipping inference warmup for listwise model during build: {name}", flush=True)
                    # Basic sanity: ensure at least one GGUF exists
                    ggufs = [f for f in os.listdir(model_dir) if f.endswith(".gguf")]
                    if not ggufs:
                        raise RuntimeError(f"Listwise model dir has no .gguf files: {model_dir}")
                    print(f"OK (downloaded): {name}", flush=True)
                else:
                    # Pairwise models: safe to actually warm up ONNX during build
                    print(f"Initializing Ranker (pairwise warmup): {name}", flush=True)
                    r = Ranker(model_name=name, cache_dir=cache_dir, max_length=max_length)
                    r.rerank(req)
                    print(f"OK (warmed): {name}", flush=True)

                last_exc = None
                break

            except Exception as e:
                last_exc = e
                print(f"FAILED: {name} (attempt {attempt}/{retries}) -> {e}", flush=True)
                if attempt < retries:
                    time.sleep(sleep_sec)

        if last_exc is not None:
            print("Warmup failed with exception:", flush=True)
            traceback.print_exc()
            raise last_exc

    print("Warmup complete.", flush=True)

except Exception:
    print("Warmup failed with exception:", flush=True)
    traceback.print_exc()
    raise
PY

# ---------- app code layer (changes often) ----------
# Only these layers rebuild when you edit code.
COPY main.py /app/main.py
COPY utils /app/utils

EXPOSE 8181
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8181"]
