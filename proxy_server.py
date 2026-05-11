import asyncio
import json
import os
import time
from typing import Iterable

import httpx
from fastapi import FastAPI, HTTPException, Request
from starlette.background import BackgroundTask
from starlette.responses import JSONResponse, StreamingResponse


LLAMA_BASE_URL = os.environ.get("LLAMA_BASE_URL", "http://127.0.0.1:8001")
LLAMA_API_KEY = os.environ.get("LLAMA_API_KEY")
MODEL_ALIAS = os.environ.get("MODEL_ALIAS", "qwen3.6-27b")
READY_TIMEOUT_SECONDS = float(os.environ.get("READY_TIMEOUT_SECONDS", "1800"))
DEFAULT_ENABLE_THINKING = os.environ.get("DEFAULT_ENABLE_THINKING", "false").lower() == "true"

app = FastAPI()


def _check_auth(request: Request) -> None:
    if not LLAMA_API_KEY:
        return

    auth = request.headers.get("authorization", "")
    x_api_key = request.headers.get("x-api-key", "")
    if auth == f"Bearer {LLAMA_API_KEY}" or x_api_key == LLAMA_API_KEY:
        return

    raise HTTPException(status_code=401, detail="Missing or invalid API key")


def _headers_without_hop_by_hop(headers: Iterable[tuple[str, str]]) -> dict[str, str]:
    excluded = {
        "connection",
        "content-length",
        "host",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailers",
        "transfer-encoding",
        "upgrade",
    }
    return {key: value for key, value in headers if key.lower() not in excluded}


async def _wait_for_llama(timeout_seconds: float = READY_TIMEOUT_SECONDS) -> None:
    deadline = time.monotonic() + timeout_seconds
    headers = {}
    if LLAMA_API_KEY:
        headers["Authorization"] = f"Bearer {LLAMA_API_KEY}"

    async with httpx.AsyncClient(timeout=5.0) as client:
        while time.monotonic() < deadline:
            try:
                response = await client.get(f"{LLAMA_BASE_URL}/health", headers=headers)
                if response.status_code < 500:
                    return
            except httpx.HTTPError:
                pass
            await asyncio.sleep(2)

    raise HTTPException(status_code=503, detail="llama-server did not become ready in time")


@app.get("/health")
async def health():
    return {"status": "proxy-ok"}


@app.api_route("/warmup", methods=["GET", "POST"])
async def warmup(request: Request):
    _check_auth(request)
    await _wait_for_llama()

    headers = {"Content-Type": "application/json"}
    if LLAMA_API_KEY:
        headers["Authorization"] = f"Bearer {LLAMA_API_KEY}"

    payload = {
        "model": MODEL_ALIAS,
        "messages": [{"role": "user", "content": "warmup"}],
        "max_tokens": 1,
        "temperature": 0,
    }

    async with httpx.AsyncClient(timeout=None) as client:
        response = await client.post(
            f"{LLAMA_BASE_URL}/v1/chat/completions",
            headers=headers,
            json=payload,
        )

    if response.status_code >= 400:
        return JSONResponse(
            status_code=response.status_code,
            content={"status": "llama-ready-but-warmup-inference-failed", "detail": response.text},
        )

    return {"status": "warm", "model": MODEL_ALIAS}


@app.api_route("/v1/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"])
async def proxy_openai(path: str, request: Request):
    _check_auth(request)
    await _wait_for_llama()

    body = await request.body()
    if request.method == "POST" and path == "chat/completions":
        try:
            payload = json.loads(body)
            if "chat_template_kwargs" not in payload:
                payload["chat_template_kwargs"] = {"enable_thinking": DEFAULT_ENABLE_THINKING}
                body = json.dumps(payload).encode("utf-8")
        except json.JSONDecodeError:
            pass

    upstream_url = f"{LLAMA_BASE_URL}/v1/{path}"
    request_headers = _headers_without_hop_by_hop(request.headers.items())
    if LLAMA_API_KEY:
        request_headers["authorization"] = f"Bearer {LLAMA_API_KEY}"

    client = httpx.AsyncClient(timeout=None)
    upstream_request = client.build_request(
        request.method,
        upstream_url,
        params=request.query_params,
        headers=request_headers,
        content=body,
    )
    upstream_response = await client.send(upstream_request, stream=True)

    response_headers = _headers_without_hop_by_hop(upstream_response.headers.items())

    async def close_upstream():
        await upstream_response.aclose()
        await client.aclose()

    return StreamingResponse(
        upstream_response.aiter_raw(),
        status_code=upstream_response.status_code,
        headers=response_headers,
        background=BackgroundTask(close_upstream),
    )
