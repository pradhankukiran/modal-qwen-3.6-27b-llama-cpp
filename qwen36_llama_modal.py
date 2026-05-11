import os
import subprocess
from pathlib import Path

import modal


APP_NAME = "qwen36-27b-llama"
MODEL_REPO = os.environ.get("APP_MODEL_REPO", "froggeric/Qwen3.6-27B-MTP-GGUF")
MODEL_FILE = os.environ.get("APP_MODEL_FILE", "Qwen3.6-27B-F16-mtp.gguf")
MODEL_ALIAS = os.environ.get("APP_MODEL_ALIAS", "qwen3.6-27b")
LLAMA_CPP_COMMIT = "5d5f1b46e4f56885801c86363d4677a5f72f83af"
MODEL_DIR = Path("/models")
MODEL_PATH = MODEL_DIR / MODEL_FILE
PUBLIC_PORT = 8000
LLAMA_PORT = 8001

CTX_SIZE = os.environ.get("APP_CTX_SIZE", "8192")
SPEC_DRAFT_N_MAX = os.environ.get("APP_SPEC_DRAFT_N_MAX", "3")

app = modal.App(APP_NAME)
model_volume = modal.Volume.from_name("qwen36-27b-models", create_if_missing=True)
remote_env = {
    key: value
    for key, value in {
        "HF_TOKEN": os.environ.get("HF_TOKEN"),
        "LLAMA_API_KEY": os.environ.get("LLAMA_API_KEY"),
    }.items()
    if value
}

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.0-devel-ubuntu22.04",
        add_python="3.12",
    )
    .apt_install(
        "build-essential",
        "ca-certificates",
        "cmake",
        "curl",
        "git",
        "libcurl4-openssl-dev",
        "ninja-build",
    )
    .pip_install(
        "fastapi>=0.115.0",
        "huggingface_hub[hf_xet]>=0.28.0",
        "httpx>=0.27.0",
        "uvicorn[standard]>=0.30.0",
    )
    .run_commands(
        (
            "git clone https://github.com/ggml-org/llama.cpp.git /opt/llama.cpp && "
            "cd /opt/llama.cpp && "
            "git fetch origin pull/22673/head:mtp-pr && "
            f"git checkout {LLAMA_CPP_COMMIT}"
        ),
        (
            "cmake -S /opt/llama.cpp -B /opt/llama.cpp/build -G Ninja "
            "-DGGML_CUDA=ON "
            "-DCMAKE_CUDA_ARCHITECTURES=90 "
            "-DCMAKE_BUILD_TYPE=Release"
        ),
        "cmake --build /opt/llama.cpp/build --target llama-server llama-cli",
    )
)


def _download_model_if_missing() -> str:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    if MODEL_PATH.exists():
        return str(MODEL_PATH)

    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        repo_id=MODEL_REPO,
        filename=MODEL_FILE,
        local_dir=str(MODEL_DIR),
    )
    model_volume.commit()
    return path


@app.function(
    image=image,
    env=remote_env,
    volumes={"/models": model_volume},
    timeout=60 * 60,
    ephemeral_disk=90 * 1024,
)
def download_model():
    path = _download_model_if_missing()
    print(f"Model ready: {path}")


@app.function(
    image=image,
    env=remote_env,
    gpu="H100",
    volumes={"/models": model_volume},
    min_containers=0,
    max_containers=1,
    scaledown_window=int(os.environ.get("APP_SCALEDOWN_WINDOW_SECONDS", "300")),
    timeout=60 * 60,
    startup_timeout=60 * 30,
    ephemeral_disk=90 * 1024,
    include_source=True,
)
@modal.concurrent(max_inputs=10)
@modal.web_server(port=PUBLIC_PORT, startup_timeout=60 * 30)
def serve():
    model_path = _download_model_if_missing()

    llama_cmd = [
        "/opt/llama.cpp/build/bin/llama-server",
        "-m",
        model_path,
        "--host",
        "127.0.0.1",
        "--port",
        str(LLAMA_PORT),
        "--alias",
        MODEL_ALIAS,
        "--ctx-size",
        CTX_SIZE,
        "-ngl",
        "999",
        "--flash-attn",
        "on",
        "--spec-type",
        "mtp",
        "--spec-draft-n-max",
        SPEC_DRAFT_N_MAX,
        "--jinja",
    ]

    api_key = os.environ.get("LLAMA_API_KEY")
    if api_key:
        llama_cmd += ["--api-key", api_key]

    print("Starting llama-server:")
    safe_llama_cmd = [
        "********" if index > 0 and llama_cmd[index - 1] == "--api-key" else value
        for index, value in enumerate(llama_cmd)
    ]
    print(" ".join(safe_llama_cmd))
    subprocess.Popen(llama_cmd)

    proxy_env = os.environ.copy()
    proxy_env.update(
        {
            "LLAMA_BASE_URL": f"http://127.0.0.1:{LLAMA_PORT}",
            "MODEL_ALIAS": MODEL_ALIAS,
        }
    )
    proxy_cmd = [
        "python",
        "-m",
        "uvicorn",
        "proxy_server:app",
        "--host",
        "0.0.0.0",
        "--port",
        str(PUBLIC_PORT),
    ]
    print("Starting API proxy:")
    print(" ".join(proxy_cmd))
    subprocess.Popen(proxy_cmd, env=proxy_env)


@app.local_entrypoint()
def main():
    download_model.remote()
