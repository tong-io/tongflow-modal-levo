"""Modal download entry for levo (LeVo 2 / SongGeneration).

Run:
  modal run download.py::download

Fetches two Hugging Face repos into the shared `models` volume:
  - lglg666/SongGeneration-v2-large -> /models/SongGeneration-v2-large
      (config.yaml + model.pt, the ckpt_path)
  - lglg666/SongGeneration-Runtime -> /models/runtime
      (ckpt/ = VAE + audio tokenizers, third_party/ = demucs + Qwen2-7B)

Self-contained: do not import other local modules.
"""

from __future__ import annotations

import os
from typing import Any

import modal




_cfg: dict[str, Any] = {}

# (repo_id, local_subdir). Overridable via plugin settings `hf.repos`.
_DEFAULT_REPOS: list[tuple[str, str]] = [
    ("lglg666/SongGeneration-v2-large", "SongGeneration-v2-large"),
    ("lglg666/SongGeneration-Runtime", "runtime"),
]


def _repo_dirs() -> list[tuple[str, str]]:
    _hf = _cfg.get("hf") if isinstance(_cfg.get("hf"), dict) else {}
    repos = _hf.get("repos")
    if isinstance(repos, list) and repos:
        out: list[tuple[str, str]] = []
        for r in repos:
            if isinstance(r, dict) and r.get("repoId"):
                rid = str(r["repoId"])
                sub = str(r.get("localDir") or rid.split("/")[-1])
                out.append((rid, sub))
        if out:
            return out
    return _DEFAULT_REPOS


volume_name = str(_cfg.get("volumeName") or "models")
volume = modal.Volume.from_name(volume_name, create_if_missing=True)
model_downloader = modal.App("model_downloader")


@model_downloader.function(
    image=modal.Image.debian_slim(python_version="3.11").pip_install(
        "huggingface_hub==1.6.0"
    ),
    volumes={"/models": volume},
    timeout=3600,
)
def _download() -> None:
    from huggingface_hub import snapshot_download

    # LeVo repos are public; HF_TOKEN is only needed if they become gated.
    token = os.environ.get("HF_TOKEN") or None

    for repo_id, subdir in _repo_dirs():
        model_dir = f"/models/{subdir}"
        if os.path.exists(model_dir) and os.listdir(model_dir):
            print(f"Model already exists at {model_dir}, skipping")
            continue
        snapshot_download(
            repo_id=repo_id,
            local_dir=model_dir,
            local_dir_use_symlinks=False,
            resume_download=True,
            token=token,
        )
        print(f"Model downloaded to {model_dir}")

    volume.commit()


@model_downloader.local_entrypoint()
def download() -> None:
    _download.remote()
