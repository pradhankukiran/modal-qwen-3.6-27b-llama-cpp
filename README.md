# Qwen3.6 27B MTP on Modal with llama.cpp

Full-quality `Qwen3.6-27B-F16-mtp.gguf` served on Modal H100 through `llama.cpp` with an OpenAI-compatible API.

## Service

- Modal app: `qwen36-27b-llama`
- Production URL: `https://YOUR-WORKSPACE--qwen36-27b-llama-serve.modal.run`
- Model: `froggeric/Qwen3.6-27B-MTP-GGUF/Qwen3.6-27B-F16-mtp.gguf`
- Runtime: `llama-server` behind a FastAPI proxy
- GPU: H100 80GB
- Context window: 262,144 tokens per request
- Scale policy: `min_containers=0`, `max_containers=1`, `scaledown_window=300`
- MTP: enabled with `--spec-type mtp --spec-draft-n-max 3`
- Thinking mode: off by default in the proxy

The warmup endpoint keeps the container/model loaded for the configured scale-down window. It does not create a persistent chat session; callers should send the needed conversation history on each request.

## Endpoints

### `GET /health`

Proxy health check. This does not prove the model is loaded.

```bash
curl -L "https://YOUR-WORKSPACE--qwen36-27b-llama-serve.modal.run/health"
```

### `GET|POST /warmup`

Starts the Modal container if needed, waits for `llama-server`, and runs a one-token inference.

```bash
curl -L -sS "https://YOUR-WORKSPACE--qwen36-27b-llama-serve.modal.run/warmup" \
  -H "Authorization: Bearer YOUR_API_KEY"
```

Expected response:

```json
{"status":"warm","model":"qwen3.6-27b"}
```

### `POST /v1/chat/completions`

OpenAI-compatible chat completions endpoint.

```bash
curl -L -sS "https://YOUR-WORKSPACE--qwen36-27b-llama-serve.modal.run/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  --data '{
    "model": "qwen3.6-27b",
    "messages": [
      {"role": "user", "content": "Reply with exactly: API works"}
    ],
    "max_tokens": 64,
    "temperature": 0
  }'
```

## Auth

Set `LLAMA_API_KEY` before deploying. Clients pass the same value as:

```text
Authorization: Bearer YOUR_API_KEY
```

or:

```text
X-API-Key: YOUR_API_KEY
```

If `LLAMA_API_KEY` is not set at deploy time, the API is public.

## OpenAI SDK

```python
from openai import OpenAI

client = OpenAI(
    base_url="https://YOUR-WORKSPACE--qwen36-27b-llama-serve.modal.run/v1",
    api_key="YOUR_API_KEY",
)

response = client.chat.completions.create(
    model="qwen3.6-27b",
    messages=[{"role": "user", "content": "Hello"}],
    max_tokens=256,
    temperature=0.2,
)

print(response.choices[0].message.content)
```

## Request Params

Common supported params:

```json
{
  "model": "qwen3.6-27b",
  "messages": [],
  "max_tokens": 512,
  "temperature": 0.2,
  "top_p": 0.9,
  "top_k": 40,
  "min_p": 0.05,
  "stream": false,
  "stop": ["..."],
  "seed": 123,
  "response_format": {"type": "json_object"},
  "chat_template_kwargs": {
    "enable_thinking": false
  }
}
```

`response_format` can be used for JSON mode or JSON-schema constrained output if supported by the current `llama.cpp` build.

## Thinking Mode

The proxy injects this by default when the caller does not provide `chat_template_kwargs`:

```json
{
  "chat_template_kwargs": {
    "enable_thinking": false
  }
}
```

To enable Qwen thinking/reasoning for one request:

```json
{
  "chat_template_kwargs": {
    "enable_thinking": true
  }
}
```

When thinking is enabled, reasoning tokens consume the same output budget as normal generated tokens. Some responses may include `message.reasoning_content` in addition to or before `message.content`.

## Token Limits

The deployed context window is:

```text
262,144 total tokens per request
```

This is input plus output plus reasoning tokens:

```text
messages + generated output + thinking tokens <= 262,144
```

Examples:

- `200K` input tokens + `20K` output tokens fits.
- `250K` input tokens leaves roughly `12K` tokens for output and reasoning.
- Thinking mode can use a large portion of `max_tokens`, so allocate more output budget when `enable_thinking=true`.

## Deploy

First authenticate Modal:

```bash
modal setup
```

Set deploy-time env vars:

```bash
export LLAMA_API_KEY='replace-with-a-long-random-token'
export HF_TOKEN='optional-huggingface-token'
export APP_CTX_SIZE=262144
export APP_SPEC_DRAFT_N_MAX=3
export APP_SCALEDOWN_WINDOW_SECONDS=300
```

Download/cache the model once into a Modal Volume:

```bash
modal run qwen36_llama_modal.py
```

Deploy:

```bash
modal deploy qwen36_llama_modal.py
```

## Local Helpers

Create a local `.env` file, which is intentionally ignored by git:

```bash
LLAMA_API_KEY=replace-with-a-long-random-token
OPENAI_API_KEY=replace-with-a-long-random-token
OPENAI_BASE_URL=https://YOUR-WORKSPACE--qwen36-27b-llama-serve.modal.run
OPENAI_MODEL=qwen3.6-27b
APP_CTX_SIZE=262144
APP_SPEC_DRAFT_N_MAX=3
APP_SCALEDOWN_WINDOW_SECONDS=300
```

Then:

```bash
set -a; . ./.env; set +a
./warmup.sh
python3 client.py
```

## Cost Notes

- Deploying does not start H100 compute by itself.
- `/warmup` and `/v1/chat/completions` can start H100 compute.
- The H100 stays warm until `scaledown_window` expires with no traffic.
- Do not send continuous warmup requests unless you want the GPU kept alive.
- Full 262K context can increase memory pressure and startup/runtime cost. Lower `APP_CTX_SIZE` if needed.
