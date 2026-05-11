# Qwen3.6 27B MTP on Modal with llama.cpp

This deploys a full-quality `Qwen3.6-27B-F16-mtp.gguf` model on a Modal H100 as an OpenAI-compatible API.

## What It Runs

- Modal app: `qwen36-27b-llama`
- Model repo: `froggeric/Qwen3.6-27B-MTP-GGUF`
- Model file: `Qwen3.6-27B-F16-mtp.gguf`
- Server: `llama-server`
- Public server: FastAPI proxy in front of `llama-server`
- Warmup route: `/warmup`
- API route: `/v1/chat/completions`
- GPU: `H100`
- Autoscaling: scale to zero, max one container, keep warm for 5 minutes
- MTP serving: single parallel sequence, because this llama.cpp MTP path requires `n_parallel=1`

## 1. Download The Model Into A Modal Volume

This downloads the 54 GB model once into the `qwen36-27b-models` Modal Volume.

```bash
cd /home/kiran/modal-qwen-3.6-27b-llama-cpp
modal run qwen36_llama_modal.py
```

## 2. Deploy The API

For a private API, set an API key for `llama-server`:

```bash
export LLAMA_API_KEY='replace-with-a-long-random-token'
modal deploy qwen36_llama_modal.py
```

Modal will print a URL like:

```text
https://YOUR-WORKSPACE--qwen36-27b-llama-serve.modal.run
```

Use that as your `OPENAI_BASE_URL`.

## 3. Warm A Session

```bash
export OPENAI_BASE_URL='https://YOUR-WORKSPACE--qwen36-27b-llama-serve.modal.run'
export OPENAI_API_KEY='replace-with-a-long-random-token'
./warmup.sh
```

The `/warmup` endpoint runs in the same Modal container pool as the OpenAI-compatible `/v1/*` API. It waits for `llama-server` to become ready, then performs a tiny one-token inference so the next real API call should avoid the boot/model-load path.

The first warmup call can take a while because Modal starts the H100 container and `llama-server` loads the model. Calls after that should be warm until `APP_SCALEDOWN_WINDOW_SECONDS` passes with no traffic.

## 4. Call From Your App

```python
from openai import OpenAI

client = OpenAI(
    base_url="https://YOUR-WORKSPACE--qwen36-27b-llama-serve.modal.run/v1",
    api_key="replace-with-a-long-random-token",
)

resp = client.chat.completions.create(
    model="qwen3.6-27b",
    messages=[{"role": "user", "content": "Hello"}],
)

print(resp.choices[0].message.content)
```

Or test with the included client:

```bash
python3 -m pip install --user --break-system-packages openai
python3 client.py
```

Direct `curl` example:

```bash
curl "${OPENAI_BASE_URL%/}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer ${OPENAI_API_KEY}" \
  --data '{
    "model": "qwen3.6-27b",
    "messages": [{"role": "user", "content": "Hello"}],
    "max_tokens": 128
  }'
```

## Useful Knobs

Set these before `modal deploy` if needed:

```bash
export APP_CTX_SIZE=16384
export APP_SPEC_DRAFT_N_MAX=3
export APP_SCALEDOWN_WINDOW_SECONDS=300
```

Start with `APP_CTX_SIZE=8192`. Increase only if you need more context, because full F16 weights plus KV cache can push H100 VRAM hard.

## Cost Controls

This config intentionally uses:

- `min_containers=0`: no always-on GPU
- `max_containers=1`: no surprise horizontal scaling
- `scaledown_window=300`: keeps the model warm for a short session

Do not send warmup requests continuously unless you want the H100 to stay running.
