"""Serve serve/app.py as a public Modal web endpoint on CPU, scaling to zero when idle.

    modal deploy serve_modal.py          # create / update the endpoint
    modal app stop slm-125m-serve        # take it down entirely

Costs nothing while idle: min_containers=0, and the container shuts down SCALEDOWN_S
seconds after the last request. max_containers=1 caps the worst case at one CPU
container (~$0.12/hour). No GPU is used.
"""

from __future__ import annotations

import modal

import config

SCALEDOWN_S = 180

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.5.1", index_url="https://download.pytorch.org/whl/cpu")
    .pip_install_from_requirements("serve/requirements.txt")
    .env({"MODEL_DIR": "/model", "RAFT_DIR": "/model-raft", "TORCH_THREADS": "4", "TOKENIZERS_PARALLELISM": "false",
          "HF_HUB_DISABLE_PROGRESS_BARS": "1"})
    # Weights baked in from the public model repo: one canonical copy, no download on cold start.
    .run_commands(
        "python -c \"from huggingface_hub import snapshot_download; "
        f"snapshot_download('{config.HF_REPO}', local_dir='/model', "
        "allow_patterns=['*.json', '*.safetensors'])\"",
        # Phase 8: the grounded fine-tune for POST /answer (docs/09-raft-release.md)
        "python -c \"from huggingface_hub import snapshot_download; "
        f"snapshot_download('{config.HF_RAFT_REPO}', local_dir='/model-raft', "
        "allow_patterns=['*.json', '*.safetensors'])\"")
    .add_local_file("serve/app.py", "/root/slm_api.py")
    .add_local_python_source("config", "raft_decode")  # config: serve_modal.py imports it; raft_decode: /answer
)

app = modal.App(f"{config.PROJECT}-serve")


@app.function(image=image, cpu=2.0, memory=4096, min_containers=0, max_containers=1,
              scaledown_window=SCALEDOWN_S, timeout=300)
@modal.concurrent(max_inputs=16)
@modal.asgi_app()
def api():
    import slm_api

    return slm_api.app
