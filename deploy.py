"""Modal deploy entry for levo (LeVo 2 / SongGeneration).

Wraps the LeVo 2 open-source song-generation model (github.com/levo-demo/LeVo,
weights lglg666/SongGeneration-v2-large) as a TongFlow `gen-music` node.

Deploy:
  modal deploy deploy.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import modal
from tongflow import deploy




_cfg: dict[str, Any] = {}

# LeVo has no pip package: the repo is cloned into the image, and the model +
# runtime bundle live in the shared `models` volume (populated by download.py).
REPO_URL = "https://github.com/levo-demo/LeVo.git"
REPO_DIR = "/app/LeVo"

# Two Hugging Face repos (see download.py):
#   - the checkpoint folder (config.yaml + model.pt) => ckpt_path
#   - the runtime bundle (ckpt/ VAE+tokenizers, third_party/ demucs+Qwen2-7B)
CKPT_DIR = "/models/SongGeneration-v2-large"
RUNTIME_DIR = "/models/runtime"

# LeVo's config.yaml pins use_flash_attn_2: true, but flash-attn is fragile to
# build. Default off (LeVo supports the standard-attention path, cf.
# `--not_use_flash_attn`); flip to True only once a matching wheel is verified.
USE_FLASH_ATTN = False

_volume_name = str(_cfg.get("volumeName") or "models")
volume = modal.Volume.from_name(_volume_name, create_if_missing=True)

from tongflow.models.gen_music import GenMusicInput, GenMusicOutput
from tongflow.node_slots import NodeSlots
from tongflow.protocol import asset, asset_as_path
from tongflow.slots import node_slot


app = modal.App(Path(__file__).resolve().parent.name)

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "libsndfile1", "ffmpeg", "build-essential")
    .pip_install("tongflow==0.2.13", "fastapi[standard]")
    .run_commands(
        f"git clone {REPO_URL} {REPO_DIR}",
        # Install pinned deps, but skip flash-attn/triton (built separately or
        # unused) — mirrors the ace-step plugin's approach.
        f"grep -viE '^(flash-attn|triton)' {REPO_DIR}/requirements.txt | pip install -r /dev/stdin",
        f"pip install --no-deps -r {REPO_DIR}/requirements_nodeps.txt",
    )
    # LeVo's deps pull in an old protobuf (<3.20) that lacks
    # EnumTypeWrapper.ValueType, which crashes Modal's in-container client at
    # cold start. Pin a Modal-compatible protobuf as the final layer so it wins.
    .pip_install("protobuf==4.25.3")
)

with image.imports():
    import io
    import os
    import sys

    import numpy as np
    import soundfile as sf
    import torch
    import torchaudio
    from omegaconf import OmegaConf

# The v2-large type_info conditioner was trained with 6 extra aesthetic
# "Musicality" tokens that the public v1 codeclm code does not add, so the
# checkpoint's type_info.output_proj is [151652, 2048] vs the v1-built
# [151646, 2048]. These are exactly the tokens the maintained ComfyUI fork adds
# for v2 (the trailing '.' is already in the Qwen vocab, so 6 net-new -> 151652).
_V2_TYPE_TOKENS = (
    "['[Musicality-very-high]', '[Musicality-high]', '[Musicality-medium]', "
    "'[Musicality-low]', '[Musicality-very-low]', '[Pure-Music]', '.']"
)


def _patch_type_info_conditioner() -> None:
    """Add the v2 Musicality tokens to QwTextConditioner before codeclm imports.

    Idempotent: edits the cloned repo's conditioners.py in place so the freshly
    built type_info.output_proj matches the v2-large checkpoint and loads with
    no size mismatch. Must run before `from codeclm...` imports the module.
    """
    import os as _os

    path = _os.path.join(REPO_DIR, "codeclm/modules/conditioners.py")
    with open(path, encoding="utf-8") as f:
        src = f.read()
    if "[Musicality-very-high]" in src:
        return
    i = src.index("class QwTextConditioner")
    anchor = "        voc_size = len(self.text_tokenizer.get_vocab())"
    add = (
        f"        self.text_tokenizer.add_tokens({_V2_TYPE_TOKENS}, "
        "special_tokens=True)\n"
    )
    src = src[:i] + src[i:].replace(anchor, add + anchor, 1)
    with open(path, "w", encoding="utf-8") as f:
        f.write(src)
    print("[levo] patched QwTextConditioner with v2 Musicality tokens")


# LeVo's reference Separator.load_audio clips prompt audio to the first 10 s
# (generate.py); token-level truncation to cfg.prompt_len happens anyway, this
# just avoids encoding minutes of audio.
PROMPT_AUDIO_MAX_SEC = 10

# Generation params (from LeVo generate.py's default inference path).
_GEN_PARAMS = dict(
    duration=None,  # filled from cfg.max_dur at load time
    extend_stride=5,
    temperature=0.9,
    cfg_coef=1.5,
    top_k=50,
    top_p=0.0,
    record_tokens=True,
    record_window=50,
)


@deploy
@app.cls(
    scaledown_window=60,
    image=image,
    gpu="L40S",
    volumes={"/models": volume},
    timeout=1800,
)
class Inference:
    @modal.enter()
    def load(self):
        # LeVo's config.yaml uses relative paths (./ckpt/..., third_party/...),
        # so run from the repo root with the runtime bundle symlinked in.
        os.chdir(REPO_DIR)
        for name in ("ckpt", "third_party"):
            link = os.path.join(REPO_DIR, name)
            target = os.path.join(RUNTIME_DIR, name)
            if os.path.islink(link) or os.path.exists(link):
                if os.path.islink(link):
                    os.remove(link)
            if not os.path.exists(link):
                os.symlink(target, link)
        os.environ.setdefault("TRANSFORMERS_CACHE", f"{REPO_DIR}/third_party/hub")

        # Mirror generate.sh's PYTHONPATH additions.
        for p in (
            REPO_DIR,
            f"{REPO_DIR}/codeclm/tokenizer",
            f"{REPO_DIR}/codeclm/tokenizer/Flow1dVAE",
        ):
            if p not in sys.path:
                sys.path.insert(0, p)

        torch.backends.cudnn.enabled = False
        OmegaConf.register_new_resolver("eval", lambda x: eval(x))
        OmegaConf.register_new_resolver(
            "concat", lambda *x: [xxx for xx in x for xxx in xx]
        )
        OmegaConf.register_new_resolver("get_fname", lambda: "default")
        OmegaConf.register_new_resolver(
            "load_yaml", lambda x: list(OmegaConf.load(x))
        )

        # Add the v2 Musicality tokens before codeclm imports read the module.
        _patch_type_info_conditioner()

        from codeclm.models import CodecLM
        from codeclm.trainer.codec_song_pl import CodecLM_PL

        cfg = OmegaConf.load(os.path.join(CKPT_DIR, "config.yaml"))
        cfg.mode = "inference"
        cfg.lm.use_flash_attn_2 = USE_FLASH_ATTN
        self.sample_rate = int(cfg.sample_rate)
        self.max_dur = float(cfg.max_dur)

        model_light = CodecLM_PL(cfg, os.path.join(CKPT_DIR, "model.pt"))
        model_light = model_light.eval().cuda()
        model_light.audiolm.cfg = cfg

        self.model = CodecLM(
            name="tmp",
            lm=model_light.audiolm,
            audiotokenizer=model_light.audio_tokenizer,
            max_duration=self.max_dur,
            seperate_tokenizer=model_light.seperate_tokenizer,
        )
        params = {**_GEN_PARAMS, "duration": self.max_dur}
        self.model.set_generation_params(**params)

    def _load_prompt_wav(self, path: str) -> "torch.Tensor":
        wav, sr = torchaudio.load(path)  # [C, T]
        if sr != self.sample_rate:
            wav = torchaudio.functional.resample(wav, sr, self.sample_rate)
        max_frames = PROMPT_AUDIO_MAX_SEC * self.sample_rate
        wav = wav[:, :max_frames]
        return wav[None]  # [1, C, T]; codeclm moves it to device itself

    def _generate_raw(
        self,
        lyric: str,
        description: str | None = None,
        duration: float | None = None,
        seed: int | None = None,
        melody_wav: "torch.Tensor | None" = None,
    ) -> bytes:
        if seed is not None:
            torch.manual_seed(int(seed))
            np.random.seed(int(seed) % (2**32))

        if duration is not None:
            params = {**_GEN_PARAMS, "duration": min(float(duration), self.max_dur)}
            self.model.set_generation_params(**params)

        generate_inp = {
            "lyrics": [lyric.replace("  ", " ")],
            "descriptions": [description],
            "melody_wavs": melody_wav,
            "vocal_wavs": None,
            "bgm_wavs": None,
            "melody_is_wav": True,
        }
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            with torch.no_grad():
                tokens = self.model.generate(**generate_inp, return_tokens=True)

        with torch.no_grad():
            wav = self.model.generate_audio(tokens, chunked=True, gen_type="mixed")

        # wav: [B, C, T] -> [T, C] for soundfile.
        audio = wav[0].cpu().float().numpy().T
        buf = io.BytesIO()
        sf.write(buf, audio, self.sample_rate, format="FLAC")
        return buf.getvalue()

    @modal.method()
    @node_slot(NodeSlots.GEN_MUSIC)
    def gen_music(self, input: GenMusicInput) -> GenMusicOutput:
        lyric = input.lyrics or input.text or ""
        if not lyric.strip():
            return GenMusicOutput(success=False, error="Missing lyrics")
        try:
            melody_wav = None
            if input.ref_audio is not None:
                with asset_as_path(input.ref_audio) as ref_path:
                    melody_wav = self._load_prompt_wav(str(ref_path))
            raw = self._generate_raw(
                lyric=lyric,
                description=input.tags or None,
                duration=input.duration,
                seed=input.seed,
                melody_wav=melody_wav,
            )
        except Exception as e:
            return GenMusicOutput(success=False, error=str(e))
        return GenMusicOutput(success=True, audio=asset(raw, mime="audio/flac"))

    @modal.fastapi_endpoint(method="GET", label=f"{Path(__file__).resolve().parent.name}-serve")
    def serve(self, taskId: str = "", token: str = "", origin: str = ""):
        from fastapi.responses import StreamingResponse
        from tongflow import serve_stream_from_spec

        return StreamingResponse(
            serve_stream_from_spec(
                origin, taskId, token, __file__,
                invoke=lambda m, inp: getattr(self, m).local(inp),
            ),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Access-Control-Allow-Origin": "*"},
        )

