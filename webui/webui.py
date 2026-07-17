"""
audio.cpp WebUI (Route B) — a thin Gradio frontend that proxies to the local
audiocpp_server HTTP API.

Model loading is on demand: instead of preloading a model at server startup, this
WebUI reads models_catalog.json, lets you pick a model, and (re)starts
audiocpp_server with a single-model config only when you actually load/run it.
One model lives in VRAM at a time — picking a different model swaps it.

Uploaded files are saved by Gradio to local temp paths, which we pass to the
server as `voice_ref` / `audio` (frontend + server run on the same machine).

Just launch this (it starts the server for you):
    venv\\Scripts\\python audiocpp-portable\\webui.py

Env overrides:
    AUDIOCPP_BACKEND=gpu|cpu     which bin dir to launch the server from
                                 (default: auto — gpu when an NVIDIA driver and the
                                 gpu server build are both present, else cpu)
    AUDIOCPP_THREADS=N           ggml compute threads (default 1; cpu backend
                                 defaults to all cores minus one)
    AUDIOCPP_SERVER=http://...   talk to an already-running server instead of managing one
    AUDIOCPP_LOAD_TIMEOUT=300    seconds to wait for a model to finish loading
    AUDIOCPP_NO_BROWSER=1        don't open a browser tab
"""
import atexit
import base64
import glob
import io
import json
import logging
import os
import random
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
import warnings
import wave
from urllib.parse import urlparse

import numpy as np
import requests
import gradio as gr

# 降噪：屏蔽 Gradio 内部触发、每次请求都会刷屏的 Starlette 弃用告警。
warnings.filterwarnings("ignore", message=r".*HTTP_422_UNPROCESSABLE.*")


def _silence_proactor_connection_reset():
    """Windows: swallow the benign `ConnectionResetError [WinError 10054]` that
    asyncio's proactor prints when a browser/HTTP connection drops abruptly."""
    if sys.platform != "win32":
        return
    try:
        from asyncio.proactor_events import _ProactorBasePipeTransport
    except Exception:
        return
    _orig = _ProactorBasePipeTransport._call_connection_lost

    def _patched(self, exc):
        if isinstance(exc, ConnectionResetError):
            exc = None  # peer reset == normal close; still run the cleanup below
        try:
            return _orig(self, exc)
        except ConnectionResetError:
            # _orig's own sock.shutdown() raced a peer reset (WinError 10054),
            # which skips the rest of its cleanup — finish it here.
            sock = getattr(self, "_sock", None)
            if sock is not None:
                try:
                    sock.close()
                except OSError:
                    pass
                self._sock = None
            server = getattr(self, "_server", None)
            if server is not None:
                try:
                    server._detach()
                except Exception:
                    pass
                self._server = None
            self._called_connection_lost = True

    _ProactorBasePipeTransport._call_connection_lost = _patched


def _silence_h11_content_length_race():
    """Large uploads (e.g. a multi-minute reference wav) can have the browser
    abort/replace an in-flight preview fetch for the same file while uvicorn
    is still streaming its body; h11 then raises LocalProtocolError trying to
    close out that half-sent response. It's a benign race — verified the
    aborted request doesn't affect the server or any other request, the
    browser's follow-up fetch of the same file completes fine — but uvicorn
    logs it as a full "Exception in ASGI application" traceback per occurrence.
    Drop just that one exception type from uvicorn's logger instead of hiding
    all uvicorn.error output."""
    try:
        from h11 import LocalProtocolError
    except Exception:
        return

    class _DropContentLengthRace(logging.Filter):
        def filter(self, record):
            exc = record.exc_info[1] if record.exc_info else None
            msg = str(exc or "")
            if isinstance(exc, LocalProtocolError) and "declared Content-Length" in msg:
                return False
            # uvicorn with httptools raises these RuntimeErrors on the same
            # preview race: "shorter" when the browser aborts/replaces the
            # fetch, "longer" when gradio sized Content-Length off a large
            # upload still being written to disk and the file grew mid-stream.
            # The request is already dead / the browser refetches; do not spam
            # a full ASGI traceback for either direction.
            if isinstance(exc, RuntimeError) and "than Content-Length" in msg:
                return False
            return True

    logging.getLogger("uvicorn.error").addFilter(_DropContentLengthRace())


def _patch_gradio_render_config_race():
    """Gradio 6.19 上游竞态（gradio-app/gradio#9991 只冻结了 blocks、漏了 fns）：
    本页有 7 个 @gr.render 动态参数区，页面加载/模型切换时多个渲染事件并发，
    一个事件在 get_config 里遍历 session 的 blocks/fns 字典的同时，另一个渲染
    正往里注册组件，偶发 "RuntimeError: dictionary changed size during
    iteration"（前端表现为该次渲染丢失/报错）。get_config 是纯只读操作，
    撞上竞态时稍等重读即可收敛。"""
    try:
        BlocksConfig = gr.blocks.BlocksConfig
        orig = BlocksConfig.get_config
    except AttributeError:
        return

    def _get_config_retry(self, renderable=None):
        for _ in range(10):
            try:
                return orig(self, renderable)
            except RuntimeError:
                time.sleep(0.02)
        return orig(self, renderable)

    BlocksConfig.get_config = _get_config_retry


_silence_proactor_connection_reset()
_silence_h11_content_length_race()
_patch_gradio_render_config_race()

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)

# WebUI-local working dirs, kept next to this file and created on startup.
CONFIG_DIR = os.path.join(HERE, "configs")
OUTPUT_DIR = os.path.join(HERE, "output")
VOICE_DIR = os.path.join(HERE, "voice")
LOG_DIR = os.path.join(HERE, "logs")
for _d in (CONFIG_DIR, OUTPUT_DIR, VOICE_DIR, LOG_DIR):
    os.makedirs(_d, exist_ok=True)

PROMPTS_DIR = VOICE_DIR                                    # built-in / reference voices
CATALOG_PATH = os.path.join(CONFIG_DIR, "models_catalog.json")
MODEL_PARAMS_PATH = os.path.join(CONFIG_DIR, "model_params.json")


# Executable/name conventions differ per OS: Windows binaries carry a .exe suffix, POSIX ones don't.
EXE_SUFFIX = ".exe" if os.name == "nt" else ""
SERVER_EXE_NAME = "audiocpp_server" + EXE_SUFFIX
# The standalone server launcher, named only in messages telling the user which
# script may be holding the port.
SERVER_LAUNCHER = "run_server.bat" if os.name == "nt" else "run_server.sh"


def _cmake_cache_backend(bin_dir):
    """gpu/cpu for a build tree whose directory name doesn't say, read from its
    CMakeCache.txt (GGML_CUDA:BOOL=ON). Returns None when there is no cache to
    read — an installed tree, or a layout we don't recognize."""
    cache = os.path.join(os.path.dirname(bin_dir), "CMakeCache.txt")
    try:
        with open(cache, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                if line.startswith("GGML_CUDA:"):
                    return "gpu" if line.rstrip().upper().endswith("ON") else "cpu"
    except OSError:
        return None
    return "cpu"


def _discover_dev_bin_dirs():
    """Locate from-source build outputs, newest-wins per backend.

    Two layouts are in the wild. The one README documents is
    build/<os>-<backend>-<type>/bin (windows-cuda-release, linux-cpu-release, …),
    where the backend is in the directory name. A plain `cmake -B build` instead
    lands in build/bin and says nothing about the backend, so that one is
    classified from its CMakeCache."""
    out = {}
    for backend, keyword in (("gpu", "*cuda*"), ("cpu", "*cpu*")):
        hits = sorted(d for d in glob.glob(os.path.join(PROJECT_ROOT, "build", keyword, "bin"))
                      if os.path.isfile(os.path.join(d, SERVER_EXE_NAME)))
        if hits:
            out[backend] = hits[-1]
    plain = os.path.join(PROJECT_ROOT, "build", "bin")
    if os.path.isfile(os.path.join(plain, SERVER_EXE_NAME)):
        backend = _cmake_cache_backend(plain)
        # Only fill a backend the named-directory scan didn't already find, so an
        # explicit build/linux-cuda-release still wins over a stale plain build/.
        if backend and backend not in out:
            out[backend] = plain
    return out


DEV_BIN_DIRS = _discover_dev_bin_dirs()


def _dev_server_exe(backend):
    d = DEV_BIN_DIRS.get(backend)
    return os.path.join(d, SERVER_EXE_NAME) if d else ""


def _find_bundle_root():
    """Locate the root that holds models/ (and, when packaged, cpu/ gpu/ tools/).

    Three layouts: a from-source dev tree (binaries under build/, models under
    PROJECT_ROOT/models), the packaged bundle next to webui/, and webui/ copied
    under the bundle for distribution. Override with AUDIOCPP_BUNDLE."""
    env = os.environ.get("AUDIOCPP_BUNDLE")
    if env:
        return env
    for c in (HERE,                                         # webui.py directly in the bundle
              PROJECT_ROOT,                                 # webui/ shipped under the bundle
              os.path.join(PROJECT_ROOT, "audiocpp-portable")):
        if os.path.isdir(os.path.join(c, "gpu")) or os.path.isdir(os.path.join(c, "cpu")):
            return c
    if any(os.path.isfile(_dev_server_exe(b)) for b in DEV_BIN_DIRS):
        return PROJECT_ROOT
    return os.path.join(PROJECT_ROOT, "audiocpp-portable")


BUNDLE_ROOT = _find_bundle_root()

def _detect_backend():
    """Which bundle build (gpu/ or cpu/) to launch. AUDIOCPP_BACKEND=gpu|cuda|cpu
    wins; otherwise auto-detect: gpu when an NVIDIA driver AND the gpu server
    build are both present, else cpu (the server runs fine on the cpu backend,
    just slower and with lower model coverage)."""
    env = os.environ.get("AUDIOCPP_BACKEND", "").strip().lower()
    if env in ("gpu", "cuda"):
        return "gpu"
    if env:
        return env
    if os.name == "nt":
        has_nvidia = os.path.isfile(os.path.join(
            os.environ.get("SystemRoot", r"C:\Windows"), "System32", "nvcuda.dll"))
    else:
        has_nvidia = shutil.which("nvidia-smi") is not None
    if has_nvidia and (os.path.isfile(os.path.join(BUNDLE_ROOT, "gpu", SERVER_EXE_NAME))
                       or os.path.isfile(_dev_server_exe("gpu"))):
        return "gpu"
    if (os.path.isfile(os.path.join(BUNDLE_ROOT, "cpu", SERVER_EXE_NAME))
            or os.path.isfile(_dev_server_exe("cpu"))):
        return "cpu"
    return "gpu"


BACKEND = _detect_backend()
# The server's own backend name: bin dirs are gpu/ vs cpu/, but the server takes
# cuda|cpu|vulkan|metal, and its config default is "cuda" — which a CPU-only
# build rejects at startup, so the temp config must always spell it out.
SERVER_BACKEND = "cuda" if BACKEND == "gpu" else BACKEND
SERVER_EXE = os.path.join(BUNDLE_ROOT, BACKEND, SERVER_EXE_NAME)
if not os.path.isfile(SERVER_EXE) and os.path.isfile(_dev_server_exe(BACKEND)):
    SERVER_EXE = _dev_server_exe(BACKEND)
LOG_PATH = os.path.join(LOG_DIR, "audiocpp_server_webui.log")
LOAD_TIMEOUT = int(os.environ.get("AUDIOCPP_LOAD_TIMEOUT", "300"))

# model_manager.py is used to download not-yet-installed models in the background.
MODELS_ROOT = os.path.join(BUNDLE_ROOT, "models")


# --- UI language -------------------------------------------------------------
# Strings live in locales/<lang>.json, one file per language. en.json is the
# source of truth and the fallback for any key a translation is missing, so a
# partial translation degrades to English rather than to a raw key. Adding a
# language means dropping in a new JSON file — no code change here.
#
# Select with AUDIOCPP_LANG (e.g. AUDIOCPP_LANG=zh). The Blocks UI is built once
# at import, so the language is fixed for the life of the process.
LOCALE_DIR = os.path.join(HERE, "locales")
DEFAULT_LANG = "en"


def available_langs():
    """Language codes with a locales/<code>.json file, sorted."""
    try:
        return sorted(f[:-5] for f in os.listdir(LOCALE_DIR) if f.endswith(".json"))
    except OSError:
        return []


def _load_locale(lang):
    try:
        with open(os.path.join(LOCALE_DIR, lang + ".json"), encoding="utf-8") as fh:
            return json.load(fh)
    except FileNotFoundError:
        return {}
    except (OSError, ValueError) as exc:
        # A broken translation must not take the UI down: warn and fall back.
        logging.warning("ignoring locale %s: %s", lang, exc)
        return {}


UI_LANG = os.environ.get("AUDIOCPP_LANG", DEFAULT_LANG).strip().lower() or DEFAULT_LANG
_FALLBACK_STRINGS = _load_locale(DEFAULT_LANG)
_STRINGS = _FALLBACK_STRINGS if UI_LANG == DEFAULT_LANG else _load_locale(UI_LANG)
if not _STRINGS:
    if UI_LANG != DEFAULT_LANG:
        logging.warning("no strings for AUDIOCPP_LANG=%s; using %s (available: %s)",
                        UI_LANG, DEFAULT_LANG, ", ".join(available_langs()) or "none")
    UI_LANG = DEFAULT_LANG
    _STRINGS = _FALLBACK_STRINGS


def set_language(lang):
    """Switch the active language, rebuilding everything derived from t().

    The language picker calls this. It is process-wide, not per browser session:
    Gradio builds one Blocks object shared by every connection, so there is nowhere
    to hang per-session labels. That is fine for a local single-user tool.

    Returns the language actually in effect (the request is ignored if the locale
    has no strings), so callers can reflect reality back into the picker.
    """
    global UI_LANG, _STRINGS
    lang = (lang or DEFAULT_LANG).strip().lower()
    strings = _FALLBACK_STRINGS if lang == DEFAULT_LANG else _load_locale(lang)
    if not strings:
        logging.warning("no strings for language %s; staying on %s", lang, UI_LANG)
        return UI_LANG
    UI_LANG, _STRINGS = lang, strings
    _rebuild_localized_tables()
    return UI_LANG


def t_opt(key):
    """Localized string for `key`, or None when no locale defines it.

    Unlike t(), a missing key is not an error here: catalog entries carry their own
    English text and only *optionally* have a translation, so callers use this to
    mean "translate this if someone has, otherwise keep what the data said"."""
    s = _STRINGS.get(key)
    return s if s is not None else _FALLBACK_STRINGS.get(key)


def t(key, **fmt):
    """Localized string for `key`, falling back to English and then the key itself.

    Keyword args are applied with str.format, so a translation can reorder or drop
    placeholders. A translation with a bad placeholder falls back rather than raising.
    """
    s = _STRINGS.get(key)
    if s is None:
        s = _FALLBACK_STRINGS.get(key, key)
    if not fmt:
        return s
    try:
        return s.format(**fmt)
    except (KeyError, IndexError):
        fallback = _FALLBACK_STRINGS.get(key, key)
        try:
            return fallback.format(**fmt)
        except (KeyError, IndexError):
            return fallback


def _find_model_manager():
    for c in (os.environ.get("AUDIOCPP_MODEL_MANAGER"),
              os.path.join(BUNDLE_ROOT, "tools", "model_manager.py"),
              os.path.join(PROJECT_ROOT, "tools", "model_manager.py")):
        if c and os.path.isfile(c):
            return c
    return None


MODEL_MANAGER = _find_model_manager()


def _detect_vram_gb():
    """本机 NVIDIA 显卡显存总量（GB，多卡取最大）；无 nvidia-smi/无 N 卡返回 None。
    用于对照 catalog 条目的 min_vram_gb 估算值，提示“下载了也可能跑不动”。"""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5)
        vals = [float(line) for line in out.stdout.split() if line.strip()]
        return round(max(vals) / 1024, 1) if vals else None
    except Exception:
        return None


LOCAL_VRAM_GB = _detect_vram_gb()


def _vram_shortfall(entry):
    """条目估算显存超过本机显存时返回 (需要GB, 本机GB)，否则 None。"""
    if BACKEND == "cpu":
        return None  # CPU 后端跑在系统内存里，显存对照不适用
    need = entry.get("min_vram_gb")
    if need and LOCAL_VRAM_GB and float(need) > LOCAL_VRAM_GB:
        return float(need), LOCAL_VRAM_GB
    return None

# Wire values for the TTS language field: what the models expect, not display text.
# "" = model default, "Auto" = let the model detect.
LANGS = ["", "english", "chinese", "french", "german", "italian",
         "japanese", "korean", "portuguese", "russian", "spanish", "Auto"]


def tts_language_choices():
    """(label, wire value) for the TTS language picker.

    Only the label is localized; the value stays the English name the server parses.
    """
    out = []
    for value in LANGS:
        if value == "":
            out.append((t("label.model_default"), ""))
        elif value == "Auto":
            out.append((t("label.autodetect"), "Auto"))
        else:
            out.append((t("lang." + value), value))
    return out

# Which catalog task tokens each tab can drive. do_tts sends text + optional
# reference voice, which fits both plain TTS ("tts") and voice cloning ("clon").
TTS_TASKS = ("tts", "clon")
ASR_TASKS = ("asr",)
GEN_TASKS = ("gen",)   # music/SFX generation, served via the generic /v1/tasks/run route
VC_TASKS = ("vc", "svc", "s2s")           # 声音转换：源音频 + 目标音色，走 /v1/tasks/run
SEP_TASKS = ("sep",)                      # 音源分离：多轨 named_audio_outputs
ANALYZE_TASKS = ("vad", "diar", "align")  # 音频分析：segments / speaker_turns / words
VDES_TASKS = ("vdes",)                    # 声音设计：文字 + 音色描述，走 /v1/audio/speech

# Per-family behavior for the TTS tab, keyed by catalog `family`. This is how we
# cope with each model wanting different input formats / options: the shared UI
# stays simple, and each family gets its own hint, optional text transform, and
# default request options. A catalog entry can also carry its own `input_hint` /
# `default_options` to override the family profile without editing this file, and
# the "高级参数 (JSON)" box lets you pass ANY model-specific option at run time.
MODEL_PROFILES = {
    "vibevoice": {
        "input_hint": t("hint.vibevoice"),
        "wrap_speaker_script": True,
        # VibeVoice has no internal text chunking. VRAM is bounded since the
        # layerwise-prefill/gallocr decode-graph fix, so chunks no longer need to be
        # tiny; 600 chars keeps each chunk's generation inside the default
        # max_tokens=1200 budget (~1.5 frames/CJK char) and the KV capacity tier
        # <= 2048 (~7.1 GB peak on 8 GB GPUs).
        "chunk_chars": 600,
        # Advanced-parameter widgets are sent only when the user edits them, so an
        # untouched num_inference_steps control (displayed 10) silently falls back to
        # the model config's ddpm_num_inference_steps=20 server-side. Send 10 (the
        # official demo default) explicitly; widgets/JSON still override.
        "default_options": {"num_inference_steps": 10},
    },
    "voxcpm2": {
        "input_hint": t("hint.voxcpm2"),
        # 8G 4060 实测标定（2026-07-05，CLI --log + nvidia-smi 抓峰值）：
        # - 峰值 ≈ 固定基线 + audiovae 解码图。基线（权重+KV+生成图）与文本长度/
        #   max_tokens 无关：max_tokens 128/300/1200 峰值都是 7634MiB（生成会提前
        #   遇停止符，解码图按实际帧数而非 max_tokens 建）。所以 chunk_chars / max_tokens
        #   都压不动基线——真正的杠杆是**权重量化**（catalog session_options 里已设
        #   voxcpm2.weight_type=q8_0：bf16 权重 4.6G，峰值 7634→5549MiB）。
        # - audiovae 解码图仍随每段音频长度涨：容量取「≥latent_frames 的最小 2 的幂」
        #   (audiovae.cpp:724)，~1.28MB/帧。q8_0 下实测 ~53 字→cap512→6366MiB，
        #   ~106 字→cap1024→7722MiB（偏紧）。所以每段要短：60 字→cap≤512→~6.4G，
        #   留 ~1.7G 给其它占 GPU 的程序。想少接缝可上调，但注意 8G 边界。
        "chunk_chars": 60,
    },
    "qwen3_tts": {
        "input_hint": t("hint.qwen3_tts"),
    },
    "pocket_tts": {
        "input_hint": t("hint.pocket_tts"),
    },
    "chatterbox": {
        "input_hint": t("hint.chatterbox"),
        # Chatterbox validates against 2-letter ISO codes (no zh/ja/ru, no auto),
        # so the shared dropdown's friendly names are translated here; names not
        # listed are genuinely unsupported by the model and rejected up front.
        "lang_map": {
            "english": "en", "spanish": "es", "french": "fr", "german": "de",
            "italian": "it", "portuguese": "pt", "korean": "ko",
        },
    },
    "qwen3_asr": {
        # Encoder cap: max_source_positions=1500 tokens at 13 tokens/second
        # (qwen3_asr_audio_encoder_token_count) -> ~115 s of audio per request.
        # 但 8G 卡上先撞显存：thinker prefill 图随时长超线性膨胀（4060 8G 实测
        # 65s 可过、70s 要 13.2GB、75s 要 14.6GB），所以客户端按 max_input_seconds
        # =60s 在静音处切段逐段转写再拼接，顺带解决 70~115s 音频原本的 OOM。
        "input_hint": t("hint.qwen3_asr"),
        "max_input_seconds": 60,
    },
    "ace_step": {
        "input_hint": t("hint.ace_step"),
        # 原版 turbo UI 默认 shift=3.0（C++ 端默认 1.0，仅 remix/extract 路由自带 3.0）。
        # 控件只发用户改过的项，所以这里显式发送，保证 UI 显示值=实际值。
        "default_options": {"shift": 3.0},
    },
    "stable_audio": {
        "input_hint": t("hint.stable_audio"),
    },
    "heartmula": {
        "input_hint": t("hint.heartmula"),
    },
    "vevo2": {
        "input_hint": t("hint.vevo2"),
        # 8G 4060 实测标定（2026-07-04/07-05）：FM 图一次建图，序列长度 =
        # 目标音色(prompt) + 源(target) 帧数（均 50fps，见 fm.cpp:782 cond_frames =
        # prompt_frames + target_frames）。cond≈25s（源15s+参考10s）就把 8G 吃满、
        # 峰值溢出到共享显存（idle 已占 7.9G/8G）；cond≈20s 勉强、30s 必炸。所以按
        # (预算 − 参考时长) 反推每段源时长，并把参考截到 ≤ ref_max，令 cond 稳定
        # 落在预算内；各段源仍补零到等长以复用缓存图。带 target_text 的编辑类 route
        # 不适合分段（会把文本对不上），这类输入本身也放不进显存。
        # 权重加载后约占 5.5-6G，留给图的只有 ~2G；18s 源(cond≈21) 只剩 ~350MB，
        # 所以预算取 16（cond≈16，留 ~0.8-1.2G 给峰值/其它占用 GPU 的程序）。
        "vc_chunk_seconds": 15,          # 每段源时长上限（会被显存预算进一步压低）
        "vc_fm_budget_seconds": 16,      # 参考 + 每段源 的总时长预算（8G 安全线）
        "vc_ref_max_seconds": 10,        # 目标音色参考截断上限
        "vc_min_chunk_seconds": 6,       # 每段源时长下限，避免切得过碎
    },
    "seed_vc": {
        "input_hint": t("hint.seed_vc"),
    },
    "miocodec": {
        "input_hint": t("hint.miocodec"),
    },
    "htdemucs": {
        "input_hint": t("hint.htdemucs"),
    },
    "mel_band_roformer": {
        "input_hint": t("hint.mel_band_roformer"),
    },
    "silero_vad": {
        "input_hint": t("hint.silero_vad"),
    },
    "marblenet_vad": {
        "input_hint": t("hint.marblenet_vad"),
    },
    "sortformer_diar": {
        "input_hint": t("hint.sortformer_diar"),
    },
    "qwen3_forced_aligner": {
        "input_hint": t("hint.qwen3_forced_aligner"),
    },
}
# Languages Qwen3-ASR can be forced to (the model config.json's support_languages;
# the prompt uses the English name). Blank/Auto = autodetect. Other ASR families
# such as citrinet ignore the field.
#
# The wire value is always the English name — only the dropdown label is localized,
# via the "lang.<value>" keys.
QWEN3_ASR_LANG_VALUES = [
    "Chinese", "English", "Cantonese", "Japanese", "Korean", "Russian",
    "French", "German", "Spanish", "Portuguese", "Italian", "Arabic",
    "Indonesian", "Thai", "Vietnamese", "Turkish", "Hindi", "Malay",
    "Dutch", "Swedish", "Danish", "Finnish", "Polish", "Czech",
    "Filipino", "Persian", "Greek", "Romanian", "Hungarian", "Macedonian",
]
def asr_language_choices():
    """(label, wire value) for the ASR language picker, including the blank/auto row.

    The label pairs the localized name with the English name the model expects, so a
    zh user picking 中文 can see it maps to "Chinese" — but only when the two differ,
    otherwise English would render as a stutter ("Chinese Chinese").
    """
    out = [(t("label.autodetect"), "")]
    for value in QWEN3_ASR_LANG_VALUES:
        name = t("lang." + value.lower())
        out.append((name if name == value else f"{name} {value}", value))
    return out


DEFAULT_PROFILE ={"input_hint": "", "wrap_speaker_script": False, "default_options": {},
                   # Families with internal chunking handle long text fine; the client
                   # split only exists to bound each HTTP request (no 900 s timeout)
                   # and surface progress, so the budget can stay coarse.
                   "chunk_chars": 1000}

# One "Speaker N:" line (any speaker index) is enough to treat text as a script.
_SPEAKER_RE = re.compile(r"^\s*Speaker\s+\d+\s*:", re.IGNORECASE | re.MULTILINE)


def profile_for(entry):
    prof = {**DEFAULT_PROFILE, **MODEL_PROFILES.get(entry.get("family", ""), {})}
    if entry.get("input_hint"):
        prof["input_hint"] = entry["input_hint"]
    if entry.get("default_options"):
        prof["default_options"] = {**prof.get("default_options", {}), **entry["default_options"]}
    return prof


def model_hint_for(model_id):
    entry = catalog_by_id(model_id) if model_id else None
    if not entry:
        return ""
    hint = profile_for(entry)["input_hint"]
    short = _vram_shortfall(entry)
    if short:
        warn = t("warn.vram_hint_uninstalled" if not entry["installed"] else "warn.vram_hint",
                 need=f"{short[0]:g}", have=f"{short[1]:g}")
        hint = warn + ("\n\n" + hint if hint else "")
    return hint


def resolve_language(prof, language):
    """Translate the shared language dropdown into what the selected family
    expects. Most families (Qwen3-TTS, VibeVoice, …) take the UI's friendly
    names as-is, so they have no `lang_map` and the value passes through. A
    family with a restricted language set (Chatterbox: 2-letter ISO codes, no
    Chinese/Japanese/Russian, no auto-detect) supplies a `lang_map`; its names
    are converted and anything it can't do — including "Auto" — is rejected here
    with an actionable message instead of a raw server 500."""
    lang = (language or "").strip()
    lang_map = prof.get("lang_map")
    if not lang_map:
        return lang
    if not lang:
        return ""                       # 留空 = 用模型默认
    code = lang_map.get(lang.lower())
    if code is not None:
        return code
    raise gr.Error(t("err.lang_unsupported", language=language,
                     supported=" / ".join(lang_map)))


def _as_speaker_script(text):
    """Wrap plain text into `Speaker 0:` lines when it isn't already a script."""
    if _SPEAKER_RE.search(text or ""):
        return text
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    return "\n".join(f"Speaker 0: {ln}" for ln in lines) if lines else text


def _vibevoice_punctuate_script(text):
    """VibeVoice is much less stable on tiny lines without sentence punctuation."""
    out = []
    for raw in (text or "").splitlines():
        line = raw.strip()
        if not line:
            continue
        m = _SPEAKER_LINE_RE.match(line)
        if not m:
            out.append(line)
            continue
        prefix, body = m.group(1), m.group(2).strip()
        if body and body[-1] not in "。！？!?.,;；:":
            body += "。"
        out.append(f"{prefix} {body}" if body else prefix)
    return "\n".join(out) if out else text


def _vibevoice_text_max_tokens(chunk, ui_default=1200):
    """Estimate VibeVoice speech tokens from text, not reference-prompt length."""
    body = re.sub(r"(?im)^\s*Speaker\s+\d+\s*:\s*", "", chunk or "")
    cjk = sum(1 for ch in body if "\u4e00" <= ch <= "\u9fff")
    non_space = sum(1 for ch in body if not ch.isspace())
    non_cjk = max(0, non_space - cjk)
    pauses = sum(1 for ch in body if ch in "，。！？；：,.!?;:")
    estimate = int(cjk * 2.2 + non_cjk * 0.45 + pauses * 3.0 + 8)
    return max(18, min(int(ui_default), estimate))


# VibeVoice 1.5B 对超短脚本从第 1 帧起整段胡言乱语——模型级缺陷（长播客数据训练，
# 短文本 OOD），与参考音色/CFG/扩散步数/中英文/seed 全部无关（2026-07-08 A/B+ASR
# 转写实测：27 字必炸、40 字逐字正确；按上面估算公式 69 token 炸 / 87 token 过）。
# 低于阈值直接拦截；生成出来只会是垃圾，调参数救不了。
_VIBEVOICE_MIN_EST_TOKENS = 85


def _merge_short_vibevoice_tail(chunks):
    """分段时留下的过短尾段同样会胡言乱语：并回前一段。前段最多超预算几十字，
    仍在 max_tokens=1200 封顶的显存包络内。"""
    while len(chunks) > 1 and (
            _vibevoice_text_max_tokens(chunks[-1]) < _VIBEVOICE_MIN_EST_TOKENS):
        tail = chunks.pop()
        chunks[-1] = chunks[-1] + "\n" + tail
    return chunks


# --- client-side long-text chunking ------------------------------------------
# Long text is synthesized as one HTTP request per chunk and concatenated here.
# That keeps every request bounded (no 900 s timeout, works for families without
# internal chunking) and lets the UI show real per-chunk progress.
_SPEAKER_LINE_RE = re.compile(r"^\s*(Speaker\s+\d+\s*:)\s*(.*)$", re.IGNORECASE)
_SENTENCE_RE = re.compile(r"[^。！？!?；;…]*[。！？!?；;…]+|[^。！？!?；;…]+$")


def _split_long_line(line, budget):
    """Split one overlong line at sentence ends into pieces of <= budget chars,
    re-attaching its `Speaker N:` prefix (if any) to every piece."""
    m = _SPEAKER_LINE_RE.match(line)
    prefix, body = (m.group(1) + " ", m.group(2)) if m else ("", line.strip())
    pieces, cur = [], ""
    for sent in _SENTENCE_RE.findall(body):
        if cur and len(cur) + len(sent) > budget:
            pieces.append(prefix + cur)
            cur = ""
        cur += sent
    if cur:
        pieces.append(prefix + cur)
    return pieces or [line]


def _split_tts_chunks(text, budget):
    """Group non-empty lines into chunks of <= budget chars. A line is never
    split across chunks unless it alone exceeds the budget (then it is split at
    sentence boundaries). Returns a list of chunk strings."""
    units = []
    for ln in (text or "").splitlines():
        if not ln.strip():
            continue
        units.extend(_split_long_line(ln, budget) if len(ln) > budget else [ln])
    chunks, cur, cur_len = [], [], 0
    for unit in units:
        sep = 1 if cur else 0  # the "\n" join separator counts toward the budget
        if cur and cur_len + sep + len(unit) > budget:
            chunks.append("\n".join(cur))
            cur, cur_len, sep = [], 0, 0
        cur.append(unit)
        cur_len += sep + len(unit)
    if cur:
        chunks.append("\n".join(cur))
    return chunks or ([text] if (text or "").strip() else [])


def _concat_wavs(blobs, out_path, keep_ratios=None):
    """Concatenate same-format WAV byte blobs into one file at out_path.
    keep_ratios[i]：每段只保留前一部分（分段转换把源补零到等长后，
    按有效占比截掉对应输出的尾部静音）。"""
    params, frames = None, []
    for i, blob in enumerate(blobs):
        with wave.open(io.BytesIO(blob)) as w:
            fmt = (w.getnchannels(), w.getsampwidth(), w.getframerate())
            if params is None:
                params = fmt
            elif fmt != params:
                raise gr.Error(t("err.chunk_format_mismatch", fmt=fmt, params=params))
            n = w.getnframes()
            if keep_ratios is not None:
                n = max(1, min(n, int(round(n * keep_ratios[i]))))
            frames.append(w.readframes(n))
    with wave.open(out_path, "wb") as w:
        w.setnchannels(params[0])
        w.setsampwidth(params[1])
        w.setframerate(params[2])
        for data in frames:
            w.writeframes(data)


def _audio_duration_seconds(path):
    """Duration of a local audio file, or None when not measurable. wave covers
    WAV, i.e. Gradio mic recordings and the typical uploads here; other formats
    just skip the duration note instead of failing the request."""
    try:
        with wave.open(path, "rb") as w:
            rate = w.getframerate()
            return (w.getnframes() / float(rate)) if rate else None
    except Exception:
        return None


# 已转码文件缓存：gradio 的临时路径按内容哈希命名，同一上传重复运行不重复转码。
_WAV_CACHE = {}


def _find_ffmpeg():
    """转码用的 ffmpeg：随 webui 分发的 ffmpeg（Windows 下为 ffmpeg.exe）优先，其次 PATH。"""
    bundled = os.path.join(HERE, "ffmpeg" + EXE_SUFFIX)
    if os.path.exists(bundled):
        return bundled
    found = shutil.which("ffmpeg")
    if not found:
        raise gr.Error(t("err.ffmpeg_missing"))
    return found


def _ensure_ascii_path(path):
    """若路径含非 ASCII 字符，复制到纯 ASCII 临时文件。
    C++ server 在 Windows 上无法打开含中文等字符的路径。"""
    try:
        path.encode("ascii")
        return path
    except UnicodeEncodeError:
        pass
    fd, out = tempfile.mkstemp(prefix="audiocpp_asc_", suffix=".wav")
    os.close(fd)
    shutil.copy2(path, out)
    return out


def _wait_file_stable(path, timeout=8.0, interval=0.08, stable_ticks=3):
    """Wait until a just-uploaded/recorded file stops changing on disk.
    On Windows, the browser can start/restart audio preview fetches while Gradio
    is still replacing a large temp file. Waiting for a few equal size samples
    before copying avoids half-written preview files."""
    if not path:
        return False
    deadline = time.time() + timeout
    last_size = -1
    ticks = 0
    while time.time() < deadline:
        try:
            size = os.path.getsize(path)
        except OSError:
            size = -1
        if size > 0 and size == last_size:
            ticks += 1
            if ticks >= stable_ticks:
                return True
        else:
            last_size = size
            ticks = 0
        time.sleep(interval)
    return os.path.exists(path)


def _safe_audio_ext(path):
    ext = os.path.splitext(path)[1].lower()
    try:
        ext.encode("ascii")
        if 0 < len(ext) <= 8:
            return ext
    except Exception:
        pass
    return ".wav"


def _stage_upload(path, force_copy=False):
    """上传/录制完成后，把输入音频换成短 ASCII 临时名再交回控件。

    关键点：长音频即使路径本身已经是短 ASCII，也强制复制到一个新的稳定文件。
    否则在同一个 gr.Audio 控件里点 X 清空再上传长音频时，旧预览 fetch 与新预览
    fetch 容易竞态，前端波形会卡住；短音频通常因为读取很快不明显。"""
    if not path or not os.path.exists(path):
        return path

    _wait_file_stable(path)

    # 已经是我们 staging 过的文件，不要再次复制，避免 .change / .upload 回写后循环。
    base = os.path.basename(path)
    if base.startswith(("audiocpp_up_", "audiocpp_rec_", "audiocpp_asc_")):
        return path

    try:
        size = os.path.getsize(path)
    except OSError:
        size = 0

    must_copy = force_copy or size >= 4 * 1024 * 1024
    try:
        path.encode("ascii")
        ascii_short = len(path) < 180
    except UnicodeEncodeError:
        ascii_short = False
        must_copy = True

    if ascii_short and not must_copy:
        return path

    ext = _safe_audio_ext(path)
    fd, out = tempfile.mkstemp(prefix="audiocpp_up_", suffix=ext)
    os.close(fd)
    shutil.copy2(path, out)
    return out


def _stage_recording(path):
    """麦克风录音结束后总是换成一个新的稳定临时文件。"""
    return _stage_upload(path, force_copy=True)


def _ensure_wav(path, target_sr=None):
    """server 端只有 WAV 读取器（其它格式报 invalid WAV RIFF header），Gradio 上传
    的 flac/mp3/ogg/m4a 等在这里先用 ffmpeg 转成 16-bit PCM WAV 临时文件再发；
    已是 RIFF/WAVE 的原样透传。target_sr：模型 prepare() 硬校验采样率的族（音源
    分离要 44.1k）传入目标值，采样率不符的输入（含 WAV）顺带重采样。"""
    if not path:
        return path
    try:
        with open(path, "rb") as f:
            head = f.read(12)
    except OSError as e:
        raise gr.Error(t("err.audio_unreadable", path=path, error=e))
    if head[:4] == b"RIFF" and head[8:12] == b"WAVE":
        if target_sr is None:
            return _ensure_ascii_path(path)
        try:
            with wave.open(path, "rb") as w:
                if w.getframerate() == target_sr:
                    return _ensure_ascii_path(path)
        except Exception:
            pass                    # 非 PCM WAV：wave 读不了，交给 ffmpeg 重写
    key = (path, target_sr)
    cached = _WAV_CACHE.get(key)
    if cached and os.path.exists(cached):
        return cached
    ffmpeg = _find_ffmpeg()
    ext = os.path.splitext(path)[1].lower() or t("label.no_extension")
    fd, out = tempfile.mkstemp(prefix="audiocpp_in_", suffix=".wav")
    os.close(fd)
    cmd = [ffmpeg, "-y", "-v", "error", "-i", path, "-map", "0:a:0"]
    if target_sr:
        cmd += ["-ar", str(target_sr)]
    cmd += ["-c:a", "pcm_s16le", out]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0 or not os.path.getsize(out):
        try:
            os.remove(out)
        except OSError:
            pass
        err = (proc.stderr or "").strip()[-300:]
        raise gr.Error(t("err.ffmpeg_failed", ext=ext, error=err or t("err.unknown")))
    note = t("log.resampled_to", sr=target_sr) if target_sr else ""
    _ui_log(t("log.input_transcoded", name=os.path.basename(path), ext=ext, note=note))
    _WAV_CACHE[key] = out
    return out


def _to_16k_mono_wav(path, target_sr=16000):
    """VAD / 说话人分离 / 强制对齐族要求 16 kHz 单声道输入（Silero、Sortformer 对
    非 16k 直接报错），这里用 wave+numpy 把 PCM WAV 转换成 16k 单声道临时文件。
    已是 16k 单声道、或非 PCM WAV（wave 读不了）时原样透传，由 server 决定成败。"""
    try:
        with wave.open(path, "rb") as w:
            sr, ch, sw = w.getframerate(), w.getnchannels(), w.getsampwidth()
            raw = w.readframes(w.getnframes())
    except Exception:
        return path
    if sr == target_sr and ch == 1:
        return path
    if sw == 2:
        data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif sw == 4:
        data = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    elif sw == 3:
        b = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
        i24 = (b[:, 0].astype(np.int32) | (b[:, 1].astype(np.int32) << 8) |
               (b[:, 2].astype(np.int32) << 16))
        i24 -= (i24 & 0x800000) << 1  # sign-extend 24-bit
        data = i24.astype(np.float32) / 8388608.0
    elif sw == 1:
        data = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    else:
        return path
    if ch > 1:
        data = data[: len(data) // ch * ch].reshape(-1, ch).mean(axis=1)
    if sr != target_sr and len(data) > 0:
        n_out = max(1, int(round(len(data) * target_sr / sr)))
        x_old = np.arange(len(data), dtype=np.float64) / sr
        x_new = np.arange(n_out, dtype=np.float64) / target_sr
        data = np.interp(x_new, x_old, data).astype(np.float32)
    fd, out = tempfile.mkstemp(prefix="audiocpp_16k_", suffix=".wav")
    os.close(fd)
    pcm = np.clip(data * 32767.0, -32768, 32767).astype(np.int16)
    with wave.open(out, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(target_sr)
        w.writeframes(pcm.tobytes())
    return out


def _split_wav_chunks(path, max_seconds, min_search_frac=0.6, win_ms=50,
                      pad_to_max=True):
    """把 PCM WAV 切成若干段临时 wav，返回 [(路径, 有效占比)]。用于 vevo2 歌声
    转换：FM 图按整段长度一次建图，8G 卡放不下长音频；且同一 server 上不同长度
    的请求会重建图并叠加占用显存（实测 20s 成功后 17s 反要 9.9GB），而形状相同
    的请求可复用缓存图（实测显存平稳）。所以**每段都补零到恰好 max_seconds**，
    占比供拼接时截掉补零对应的尾部输出。切点选在
    [起点+max*min_search_frac, 起点+max] 区间内能量最低的 win_ms 窗口中心
    （典型是呼吸/间奏处）；读不了的（非 PCM WAV）原样返回不分段。
    pad_to_max=False：不补零（ASR 分段转写用——补出的尾部静音只会浪费编码器
    token 甚至诱发幻听，转写也不需要各段图形状一致）。"""
    try:
        with wave.open(path, "rb") as w:
            sr, ch, sw = w.getframerate(), w.getnchannels(), w.getsampwidth()
            n = w.getnframes()
            raw = w.readframes(n)
    except Exception:
        return [(path, 1.0)]            # 非 PCM WAV：不分段，交给 server
    if sw != 2 or n <= 0:
        return [(path, 1.0)]
    data = np.frombuffer(raw, dtype=np.int16)
    frames = data.reshape(-1, ch) if ch > 1 else data.reshape(-1, 1)
    mono = np.abs(frames.astype(np.float32)).mean(axis=1)
    win = max(1, int(sr * win_ms / 1000.0))
    env_len = max(1, len(mono) // win)
    env = mono[: env_len * win].reshape(env_len, win).mean(axis=1)

    max_f, lo_f = int(max_seconds * sr), int(max_seconds * min_search_frac * sr)
    spans, pos = [], 0
    while n - pos > max_f:
        w0 = min((pos + lo_f) // win, env_len - 1)
        w1 = min((pos + max_f) // win, env_len)
        wi = w0 + int(np.argmin(env[w0:w1])) if w1 > w0 else w1 - 1
        cut = min(n, wi * win + win // 2)
        spans.append((pos, cut))
        pos = cut
    if pos < n:
        spans.append((pos, n))
    outs = []
    for i, (a, b) in enumerate(spans):
        fd, out = tempfile.mkstemp(prefix=f"audiocpp_vcseg{i}_", suffix=".wav")
        os.close(fd)
        seg = frames[a:b]
        if pad_to_max and len(seg) < max_f:  # 全部段补零到等长，保证图形状一致
            pad = np.zeros((max_f - len(seg), ch), dtype=np.int16)
            seg = np.concatenate([seg, pad], axis=0)
        with wave.open(out, "wb") as ww:
            ww.setnchannels(ch)
            ww.setsampwidth(sw)
            ww.setframerate(sr)
            ww.writeframes(seg.tobytes())
        outs.append((out, (b - a) / float(max_f)))
    return outs


def _trim_wav_seconds(path, max_seconds):
    """把 PCM WAV 截到前 max_seconds 秒。vevo2 的音色参考超过约 10s 对音色几乎
    没有额外贡献，却会让 FM 图的 prompt 段变长、和源音频一起把显存吃满，所以
    转换前先截短。短于上限、或非 PCM WAV（wave 读不了）的原样返回。"""
    try:
        with wave.open(path, "rb") as w:
            sr, ch, sw = w.getframerate(), w.getnchannels(), w.getsampwidth()
            keep = int(max_seconds * sr)
            if keep <= 0 or w.getnframes() <= keep:
                return path
            raw = w.readframes(keep)
    except Exception:
        return path
    fd, out = tempfile.mkstemp(prefix="audiocpp_ref_", suffix=".wav")
    os.close(fd)
    with wave.open(out, "wb") as ww:
        ww.setnchannels(ch)
        ww.setsampwidth(sw)
        ww.setframerate(sr)
        ww.writeframes(raw)
    return out


def _parse_adv_options(raw):
    raw = (raw or "").strip()
    if not raw:
        return {}
    try:
        obj = json.loads(raw)
    except Exception as e:
        raise gr.Error(t("err.adv_json_invalid", error=e))
    if not isinstance(obj, dict):
        raise gr.Error(t("err.adv_json_not_object"))
    return obj


# Map known server-error fragments to an actionable hint key, so a raw 500 like
# "requires a session voice via --voice-ref" becomes "please upload a reference voice".
# Ordered specific -> generic; server_error() takes the FIRST match and localizes it.
ERROR_HINTS = [
    (re.compile(r"failed to allocate .{0,40}graph|out of memory|cudaMalloc", re.I),
     "hint.err.oom"),
    (re.compile(r"invalid WAV RIFF header", re.I),
     "hint.err.wav_only"),
    (re.compile(r"unsupported Chatterbox language", re.I),
     "hint.err.chatterbox_lang"),
    (re.compile(r"Stable Audio.{0,80}(English|prompt)|prompt.{0,80}(English|Stable Audio)", re.I),
     "hint.err.stable_audio_english"),
    (re.compile(r"max_source_positions", re.I),
     "hint.err.asr_too_long"),
    (re.compile(r"exceeds fixed graph capacity|session_len_sec exceeds", re.I),
     "hint.err.graph_capacity"),
    (re.compile(r"combine voice_samples|voice_samples.{0,20}voice_ref", re.I),
     "hint.err.voice_samples_conflict"),
    (re.compile(r"cached voice id", re.I),
     "hint.err.needs_voice_file"),
    (re.compile(r"no valid Speaker|Speaker\s+N", re.I),
     "hint.err.needs_speaker_script"),
    (re.compile(r"reference[-_ ]?text", re.I),
     "hint.err.needs_reference_text"),
    (re.compile(r"voice[-_ ]?ref|voice[-_ ]?id|session voice|speaker reference|"
                r"requires .{0,40}voice|requires audio", re.I),
     "hint.err.needs_voice_ref"),
]


def _extract_server_message(text):
    try:
        return json.loads(text)["error"]["message"] or text
    except Exception:
        return (text or "").strip()


def server_error(entry, status, text, extra=None):
    """Build a friendly gr.Error from a non-200 server response."""
    msg = _extract_server_message(text)
    key = next((k for pat, k in ERROR_HINTS if pat.search(msg)), None)
    parts = [f"❌ server {status}"]
    if key:
        parts.append("💡 " + t(key))
    else:
        parts[0] = t("err.server_status", status=status, message=msg[:400])
    if extra:
        parts.append(extra)
    if entry and not key:
        ih = profile_for(entry).get("input_hint")
        if ih:
            parts.append("ℹ️ " + ih)
    return gr.Error("\n\n".join(parts))


def _msg_from_error(e):
    """Human-readable text from a gr.Error/exception, for inline (non-popup) display."""
    return getattr(e, "message", None) or str(e) or t("err.unknown")


# Fallback catalog if models_catalog.json is missing/unreadable.
DEFAULT_CATALOG = {
    "host": "127.0.0.1", "port": 8080, "device": 0, "threads": 1,
    "models": [
        {"id": "qwen3-tts", "display_name": "Qwen3-TTS 0.6B (tts)",
         "family": "qwen3_tts", "path": "models/Qwen3-TTS-12Hz-0.6B-Base",
         "task": "tts", "mode": "offline"},
        {"id": "vibevoice", "display_name": "VibeVoice 1.5B (tts)",
         "family": "vibevoice", "path": "models/VibeVoice-1.5B",
         "task": "tts", "mode": "offline"},
        {"id": "qwen3-asr", "display_name": "Qwen3-ASR 0.6B (asr)",
         "family": "qwen3_asr", "path": "models/Qwen3-ASR-0.6B",
         "task": "asr", "mode": "offline"},
    ],
}


def _load_catalog():
    if os.path.isfile(CATALOG_PATH):
        try:
            with open(CATALOG_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"[webui] failed to read {CATALOG_PATH}: {e}; using defaults")
    return DEFAULT_CATALOG


def _load_model_params():
    """Per-family advanced-parameter specs for the TTS tab (configs/model_params.json).
    Returns {family: [param_spec, ...]}; a missing/broken file -> {} (no knobs shown)."""
    if os.path.isfile(MODEL_PARAMS_PATH):
        try:
            with open(MODEL_PARAMS_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"[webui] failed to read {MODEL_PARAMS_PATH}: {e}; no param controls")
    return {}


CATALOG = _load_catalog()
MODEL_PARAMS = _load_model_params()
HOST = CATALOG.get("host", "127.0.0.1")
PORT = int(CATALOG.get("port", 8080))
DEVICE = int(CATALOG.get("device", 0))
THREADS = int(os.environ.get("AUDIOCPP_THREADS") or CATALOG.get("threads", 1))
if BACKEND == "cpu" and THREADS <= 1:
    # catalog 里的 threads=1 是按 CUDA 调的（GPU 路径不吃这个值）；CPU 后端的
    # ggml 计算线程数就是它，单线程没法用 —— 默认全核减一，留一个核给 UI/系统。
    THREADS = max(1, (os.cpu_count() or 4) - 1)

# If the user points us at an existing server, keep everything consistent with it.
_ENV_SERVER = os.environ.get("AUDIOCPP_SERVER")
if _ENV_SERVER:
    _u = urlparse(_ENV_SERVER)
    HOST = _u.hostname or HOST
    PORT = _u.port or PORT
    SERVER = _ENV_SERVER.rstrip("/")
else:
    SERVER = f"http://{HOST}:{PORT}"

# --- managed server process state ------------------------------------------
_proc_lock = threading.Lock()
_server_proc = None      # subprocess.Popen we launched, or None
_loaded_id = None        # model id our managed server is serving
_loaded_session_options = None   # 随本次加载写进 server config 的 session_options


def catalog_models():
    """Catalog entries annotated with abs_path / installed / label."""
    out = []
    for m in CATALOG.get("models", []):
        rel = m.get("path", "")
        ap = rel if os.path.isabs(rel) else os.path.join(BUNDLE_ROOT, rel)
        entry = dict(m)
        entry["abs_path"] = os.path.normpath(ap).replace("\\", "/")
        entry["installed"] = os.path.exists(entry["abs_path"])
        # The catalog holds English text; a locale may override any entry's user-facing
        # strings under "catalog.<id>.<field>". Untranslated entries keep the catalog's
        # own wording, so a third-party catalog needs no locale work to stay readable.
        mid = m.get("id", "?")
        hint = t_opt(f"catalog.{mid}.input_hint")
        if hint:
            entry["input_hint"] = hint
        entry["label"] = (t_opt(f"catalog.{mid}.display_name")
                          or m.get("display_name") or mid)
        out.append(entry)
    return out


def catalog_by_id(model_id):
    for m in catalog_models():
        if m.get("id") == model_id:
            return m
    return None


def choices_for_tasks(tasks):
    """[(label, id)] for catalog models whose task is in `tasks`; missing ones flagged.
    未安装且估算显存超过本机的条目额外标注最低显存，防止白下载。"""
    out = []
    for m in catalog_models():
        if m.get("task") not in tasks:
            continue
        label = m["label"]
        if not m["installed"]:
            label += " · " + t("label.not_installed")
            short = _vram_shortfall(m)
            if short:
                label += " " + t("label.vram_estimate", need=f"{short[0]:g}")
        out.append((label, m["id"]))
    return out


def builtin_voices():
    if not os.path.isdir(PROMPTS_DIR):
        return []
    return sorted(f for f in os.listdir(PROMPTS_DIR) if f.lower().endswith(".wav"))


def _load_voice_texts():
    """Built-in voice basename -> reference transcript, parsed from
    voice/prompt_text (each line is '<basename>|<transcript>')."""
    texts = {}
    try:
        with open(os.path.join(PROMPTS_DIR, "prompt_text"), "r", encoding="utf-8") as f:
            for line in f:
                name, sep, text = line.rstrip("\n").partition("|")
                if sep:
                    texts[name.strip()] = text
    except Exception:
        pass
    return texts


def refresh_builtin_voices(current):
    """刷新按钮：重新扫描 voice/ 目录的 wav 列表，并按当前选中项重读
    prompt_text 里的参考文本；选中项已被删除时回落到 '(none)'。
    必须在同一个 handler 里连带输出 wav/参考文本——拆成 .then 链会和
    下拉更新触发的 .change 并发执行，撞 Gradio get_config 的竞态
    （RuntimeError: dictionary changed size during iteration）。"""
    choices = ["(none)"] + builtin_voices()
    if current not in choices:
        current = "(none)"
    wav, ref = on_builtin_voice_change(current)
    return gr.update(choices=choices, value=current), wav, ref


def on_builtin_voice_change(name):
    """Selecting a built-in voice mirrors its wav into the upload widget and
    fills the matching reference text; '(none)' clears both."""
    if not name or name == "(none)":
        return None, ""
    path = os.path.join(PROMPTS_DIR, name)
    ref = _load_voice_texts().get(os.path.splitext(name)[0], "")
    return (path if os.path.isfile(path) else None), ref


# --- config-driven advanced-parameter controls (TTS tab) -------------------
def params_for(model_id):
    """Advanced-parameter specs for a model, looked up by its catalog family."""
    entry = catalog_by_id(model_id) if model_id else None
    if not entry:
        return []
    specs = MODEL_PARAMS.get(entry.get("family", ""), [])
    return specs if isinstance(specs, list) else []


def _make_param_component(p):
    """Build one Gradio control from a spec (type: slider|number|bool|text|choice).

    interactive=True is forced: inside @gr.render a control that is only wired to
    its own .change handler is otherwise inferred as output-only (read-only)."""
    t = p.get("type", "number")
    label = p.get("label", p.get("name", ""))
    info = p.get("info")
    if t == "bool":
        return gr.Checkbox(label=label, info=info, value=bool(p.get("default", False)),
                           interactive=True)
    if t == "text":
        return gr.Textbox(label=label, info=info, value=p.get("default", ""),
                          placeholder=p.get("placeholder", ""),
                          lines=int(p.get("lines", 1)), interactive=True)
    if t == "choice":
        return gr.Dropdown(label=label, info=info, choices=p.get("choices", []),
                           value=p.get("default"), interactive=True)
    if t == "slider":
        return gr.Slider(label=label, info=info,
                         minimum=p.get("minimum", 0), maximum=p.get("maximum", 1),
                         step=p.get("step", 0.01), value=p.get("default", 0),
                         interactive=True)
    return gr.Number(label=label, info=info, value=p.get("default"),
                     minimum=p.get("minimum"), maximum=p.get("maximum"),
                     step=p.get("step"), precision=p.get("precision"),
                     interactive=True)


def _adv_updater(name):
    """change-handler that writes one control's value into the shared advanced
    options state dict (keyed by the option name)."""
    def _fn(state, value):
        state = dict(state or {})
        state[name] = value
        return state
    return _fn


# --- server lifecycle ------------------------------------------------------
def _port_open(host, port, timeout=0.5):
    try:
        with socket.create_connection((host, int(port)), timeout=timeout):
            return True
    except OSError:
        return False


def server_alive():
    try:
        requests.get(f"{SERVER}/health", timeout=3).raise_for_status()
        return True
    except Exception:
        return False


def loaded_ids():
    try:
        r = requests.get(f"{SERVER}/v1/models", timeout=5)
        r.raise_for_status()
        return [m.get("id") for m in r.json().get("data", [])]
    except Exception:
        return []


def _write_temp_config(entry):
    model = {
        "id": entry["id"],
        "family": entry["family"],
        "path": entry["abs_path"],
        "task": entry.get("task", "tts"),
        "mode": entry.get("mode", "offline"),
    }
    for key in ("config", "weight", "load_options", "session_options"):
        if entry.get(key) is not None:
            model[key] = entry[key]
    cfg = {"host": HOST, "port": PORT, "backend": SERVER_BACKEND, "device": DEVICE,
           "threads": THREADS, "models": [model]}
    fd, path = tempfile.mkstemp(prefix="audiocpp_webui_cfg_", suffix=".json")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(cfg, f)
    return path


def _stop_server():
    global _server_proc, _loaded_id, _loaded_session_options
    _loaded_session_options = None
    proc, _server_proc, _loaded_id = _server_proc, None, None
    if proc is not None and proc.poll() is None:
        try:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)
        except Exception:
            pass
    for _ in range(24):  # let the OS release the port
        if not _port_open(HOST, PORT):
            break
        time.sleep(0.25)


def _read_tail(path, n=30, max_bytes=65536):
    """Last n non-empty lines of a file. Reads only the file's tail, and treats
    \\r as a line break so tqdm-style progress (one giant \\r-line) shows its
    latest state instead of nothing."""
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - max_bytes))
            data = f.read().decode("utf-8", errors="replace")
        lines = [ln for ln in data.replace("\r\n", "\n").replace("\r", "\n").split("\n")
                 if ln.strip()]
        return "\n".join(lines[-n:]).strip()
    except Exception:
        return ""


def _log_tail(n=30):
    return _read_tail(LOG_PATH, n)


# One shared append handle for the WebUI log: the pump thread (server output)
# and _ui_log (webui-side request events) both write through it, so lines
# interleave correctly instead of two handles overwriting each other.
_log_lock = threading.Lock()
_log_fh = None


def _open_log_file(truncate=False):
    global _log_fh
    with _log_lock:
        if _log_fh is not None:
            try:
                _log_fh.close()
            except Exception:
                pass
        _log_fh = open(LOG_PATH, "w" if truncate else "a",
                       encoding="utf-8", errors="replace")


def _log_write(text):
    global _log_fh
    with _log_lock:
        if _log_fh is None:
            try:
                _log_fh = open(LOG_PATH, "a", encoding="utf-8", errors="replace")
            except Exception:
                return
        try:
            _log_fh.write(text)
            _log_fh.flush()
        except Exception:
            pass


def _ts():
    return time.strftime("%H:%M:%S")


def _emit_log_line(text):
    """One already-formatted line to BOTH the console and the WebUI log file."""
    try:
        sys.stdout.write(text)
        sys.stdout.flush()
    except Exception:
        pass
    _log_write(text)


def _ui_log(msg):
    """Timestamped webui-side event (request start/finish, model load, ...) so
    the console/log show when things began and ended, not just server spam."""
    _emit_log_line(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [webui] {msg}\n")


def _pump_server_output(proc):
    """Tee the server's combined stdout/stderr to console + log file, prefixing
    every line with a timestamp and collapsing consecutive duplicate lines
    (e.g. the repeated `CUDA graph warmup complete`) into a periodic counter."""
    last, repeats = None, 0
    try:
        for line in proc.stdout:
            if line == last:
                repeats += 1
                if repeats % 50 == 0:
                    _emit_log_line(f"[{_ts()}]   ... " + t("log.repeated_so_far", n=repeats) + "\n")
                continue
            if repeats:
                _emit_log_line(f"[{_ts()}]   ... (" + t("log.repeated_total", n=repeats) + ")\n")
            last, repeats = line, 0
            _emit_log_line(f"[{_ts()}] {line}")
    except Exception:
        pass
    finally:
        if repeats:
            _emit_log_line(f"[{_ts()}]   ... (" + t("log.repeated_total", n=repeats) + ")\n")


def _start_server(entry):
    global _server_proc, _loaded_id
    if not os.path.isfile(SERVER_EXE):
        raise gr.Error(t("err.server_exe_missing", path=SERVER_EXE))
    cfg = _write_temp_config(entry)
    flags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
    _open_log_file(truncate=True)
    # --log 会打开 engine 的 [TRACE]/[TIMING] 调试输出，日常太吵，默认关闭；
    # 排查推理问题时设 AUDIOCPP_SERVER_DEBUG=1 再启动 webui。
    cmd = [SERVER_EXE, "--config", cfg, "--host", HOST, "--port", str(PORT)]
    if os.environ.get("AUDIOCPP_SERVER_DEBUG") == "1":
        cmd.append("--log")
    _server_proc = subprocess.Popen(
        cmd,
        cwd=BUNDLE_ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        creationflags=flags, text=True, encoding="utf-8", errors="replace", bufsize=1,
    )
    _loaded_id = entry["id"]
    extra = f", threads={THREADS}" if SERVER_BACKEND == "cpu" else ""
    _ui_log(t("log.server_starting", backend=SERVER_BACKEND, extra=extra, label=entry["label"]))
    threading.Thread(target=_pump_server_output, args=(_server_proc,),
                     daemon=True).start()


def _wait_health(timeout):
    start = time.time()
    while time.time() - start < timeout:
        if _server_proc is not None and _server_proc.poll() is not None:
            return False  # process exited before becoming healthy
        if server_alive():
            return True
        time.sleep(0.5)
    return False


def ensure_model_loaded(model_id, expect_tasks=None, session_options=None):
    """(Re)start the server so `model_id` is loaded. Returns a status string.
    session_options：额外写进本次 server config 的 session_options（string→string，
    如 sortformer 的 session_len_sec）；与上次加载不一致时会重启重载。"""
    global _loaded_session_options
    if not model_id:
        raise gr.Error(t("err.no_model_selected"))
    entry = catalog_by_id(model_id)
    if entry is None:
        raise gr.Error(t("err.model_id_unknown", model_id=model_id))
    if not entry["installed"]:
        raise gr.Error(t("err.model_not_installed", path=entry["abs_path"]))
    if expect_tasks and entry.get("task") not in expect_tasks:
        raise gr.Error(t("err.model_wrong_task", model_id=model_id,
                         task=entry.get("task"), expected="/".join(expect_tasks)))

    with _proc_lock:
        managed_alive = _server_proc is not None and _server_proc.poll() is None
        if (managed_alive and _loaded_id == model_id and server_alive()
                and (session_options or {}) == (_loaded_session_options or {})):
            return t("msg.model_loaded", label=entry["label"])

        if not managed_alive and server_alive():
            # A server we didn't launch is holding the port.
            if session_options:
                raise gr.Error(t("err.external_server_session", host=HOST, port=PORT,
                                 options=session_options))
            if model_id in loaded_ids():
                return t("msg.reusing_external_server", label=entry["label"])
            raise gr.Error(t("err.external_server_port", host=HOST, port=PORT,
                             launcher=SERVER_LAUNCHER))

        _stop_server()
        if session_options:
            entry = dict(entry)
            entry["session_options"] = {
                **(entry.get("session_options") or {}), **session_options}
        t0 = time.time()
        _start_server(entry)
        _loaded_session_options = dict(session_options) if session_options else None
        if not _wait_health(LOAD_TIMEOUT):
            tail = _log_tail()
            _stop_server()
            _ui_log(t("log.model_load_failed", label=entry["label"], timeout=LOAD_TIMEOUT))
            raise gr.Error(t("err.model_load_failed", label=entry["label"],
                             timeout=LOAD_TIMEOUT, tail=tail))
        _ui_log(t("log.model_loaded", label=entry["label"], secs=f"{time.time() - t0:.1f}"))
        return t("msg.model_loaded", label=entry["label"])


def unload_model():
    """停止本 WebUI 启动的 server，释放全部显存（权重+常驻计算图缓冲）。
    下次生成/转写时 ensure_model_loaded 会自动重启重载（实测 ~5s），转写速度
    本身不受影响（计算图本来就按每次请求的音频长度分配/复用）。"""
    with _proc_lock:
        managed_alive = _server_proc is not None and _server_proc.poll() is None
        if managed_alive:
            label = _loaded_id or "(unknown)"
            _stop_server()
            _ui_log(t("log.model_unloaded", label=label))
            return t("msg.unloaded"), server_status()
        if server_alive():
            return t("msg.unload_external", launcher=SERVER_LAUNCHER), server_status()
        return t("msg.unload_not_running"), server_status()


def server_status():
    if server_alive():
        ids = ", ".join(loaded_ids()) or "(none)"
        return f"✅ server @ {SERVER} · backend={BACKEND} · model id={ids}"
    return t("msg.server_not_running", server=SERVER, load=t("btn.load"))


def _api_usage_md():
    """Third-party call instructions for the collapsible section under the status
    line. Endpoints/fields track app/server/README.md; the URL comes from the
    runtime SERVER so it cannot drift from AUDIOCPP_SERVER / the catalog config."""
    return t("api.usage_md", server=SERVER, launcher=SERVER_LAUNCHER,
             load=t("btn.load"))


# --- background model downloads (via tools/model_manager.py) ----------------
_dl_lock = threading.Lock()
_downloads = {}  # model_id -> {"proc": Popen, "log": path}


def _dl_log_path(model_id):
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", model_id)
    return os.path.join(LOG_DIR, f"download_{safe}.log")


def _dir_size_bytes(path):
    total = 0
    for root, _dirs, files in os.walk(path):
        for name in files:
            try:
                total += os.path.getsize(os.path.join(root, name))
            except OSError:
                pass
    return total


def _fmt_bytes(n):
    return f"{n / 1e9:.2f} GB" if n >= 1e9 else f"{n / 1e6:.1f} MB"


def _download_progress_note(entry):
    """Bytes already on disk for a running download. model_manager stages into
    models/.engine_model_staging/<target>.partial/ and renames on completion;
    fall back to the whole staging root for packages with composite targets."""
    staging_root = os.path.join(MODELS_ROOT, ".engine_model_staging")
    base = os.path.basename(entry.get("path", "").rstrip("/\\"))
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", base)
    staging = os.path.join(staging_root, safe + ".partial")
    probe = staging if os.path.isdir(staging) else staging_root
    if not os.path.isdir(probe):
        return t("msg.dl_no_data_yet")
    return t("msg.dl_downloaded", size=_fmt_bytes(_dir_size_bytes(probe)))


def hf_token_present():
    """True if model_manager will find an HF token (env or cached login)."""
    if os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"):
        return True
    return os.path.isfile(os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "token"))


def download_model(model_id, hf_token="", proxy=""):
    """Kick off `model_manager.py install <download_id>` in the background."""
    if not model_id:
        return "❌ " + t("err.no_model_selected")
    entry = catalog_by_id(model_id)
    if entry is None:
        return "❌ " + t("err.model_id_unknown", model_id=model_id)
    if entry["installed"]:
        return t("msg.dl_already_installed", label=entry["label"])
    dl_id = entry.get("download_id")
    if not dl_id:
        return t("msg.dl_no_download_id", label=entry["label"])
    if MODEL_MANAGER is None:
        return t("err.model_manager_missing")

    # Pass a token to the child so gated/private HF repos don't 401.
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"    # progress lines land in the log immediately
    env["PYTHONIOENCODING"] = "utf-8"
    tok = (hf_token or "").strip()
    if tok:
        env["HF_TOKEN"] = tok
        env["HUGGING_FACE_HUB_TOKEN"] = tok
    # Route the child's downloads through a proxy (urllib reads these env vars).
    px = (proxy or "").strip()
    proxy_note = ""
    if px:
        for var in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
            env[var] = px
        proxy_note = t("msg.dl_via_proxy", proxy=px) + "\n\n"
    warn = "" if (tok or hf_token_present()) else (t("warn.no_hf_token") + "\n\n")
    short = _vram_shortfall(entry)
    if short:
        warn = t("warn.vram_before_download", need=f"{short[0]:g}",
                 have=f"{short[1]:g}") + "\n\n" + warn

    with _dl_lock:
        rec = _downloads.get(model_id)
        if rec and rec["proc"].poll() is None:
            return (t("msg.dl_already_running", label=entry["label"])
                    + f"\n```\n{_read_tail(rec['log'])}\n```")
        log = _dl_log_path(model_id)
        logf = open(log, "w", encoding="utf-8", errors="replace")
        proc = subprocess.Popen(
            [sys.executable, "-u", MODEL_MANAGER, "install", dl_id,
             "--models-root", MODELS_ROOT, "--overwrite"],
            cwd=PROJECT_ROOT, stdout=logf, stderr=subprocess.STDOUT, env=env)
        _downloads[model_id] = {"proc": proc, "log": log}
    _ui_log(t("log.dl_started", label=entry["label"], dl_id=dl_id, log=log))
    return warn + proxy_note + t("msg.dl_started", label=entry["label"], dl_id=dl_id,
                                 log=log, refresh=t("btn.refresh"))


def download_status(model_id):
    entry = catalog_by_id(model_id) if model_id else None
    if entry is None:
        return ""
    if entry["installed"]:
        return t("msg.dl_installed", label=entry["label"])
    rec = _downloads.get(model_id)
    if rec is None:
        return t("msg.dl_not_started", label=entry["label"])
    code = rec["proc"].poll()
    tail = _read_tail(rec["log"], n=12)
    if code is None:
        return (t("msg.dl_running", label=entry["label"],
                  progress=_download_progress_note(entry), ts=_ts())
                + f"\n```\n{tail}\n```")
    if not rec.get("reported"):
        rec["reported"] = True
        _ui_log(t("log.dl_exited", label=entry["label"], code=code))
    if code == 0:
        return (t("msg.dl_done", label=entry["label"], refresh=t("btn.refresh"))
                + f"\n```\n{tail}\n```")
    return t("msg.dl_failed", label=entry["label"], code=code) + f"\n```\n{tail}\n```"


def _download_running(model_id):
    rec = _downloads.get(model_id) if model_id else None
    return rec is not None and rec["proc"].poll() is None


def download_start(model_id, hf_token="", proxy=""):
    """Click handler: kick off the download and arm the auto-refresh timer."""
    msg = download_model(model_id, hf_token, proxy)
    return msg, gr.Timer(active=_download_running(model_id))


def download_status_tick(model_id):
    """Timer tick: refresh status; stop the timer once the download is idle."""
    return download_status(model_id), gr.Timer(active=_download_running(model_id))


# --- task handlers ---------------------------------------------------------
# Task handlers return (output, message): the reminder/status message is shown
# inline under the output widget instead of as a Gradio popup card.
def _merged_options(prof, adv_values, adv_options):
    """请求 options 合并：family 默认值 -> 生成控件（仅用户改过的项）-> JSON 兜底框。"""
    options = dict(prof.get("default_options", {}))
    if isinstance(adv_values, dict):
        options.update({k: v for k, v in adv_values.items()
                        if v is not None and v != ""})
    options.update(_parse_adv_options(adv_options))
    return options


def _run_task(entry, model, req, timeout, log_label):
    """POST /v1/tasks/run（通用任务路由）并返回响应 JSON；
    连接失败 / 非 200 统一转成带提示的 gr.Error。"""
    try:
        r = requests.post(f"{SERVER}/v1/tasks/run",
                          json={"model": model, "request": req}, timeout=timeout)
    except requests.RequestException as e:
        _ui_log(t("log.task_failed_unreachable", task=log_label))
        raise gr.Error(t("err.server_unreachable", server=SERVER, error=e, load=t("btn.load")))
    if r.status_code != 200:
        _ui_log(t("log.task_failed_status", task=log_label, status=r.status_code))
        raise server_error(entry, r.status_code, r.text)
    return r.json()


def _resolve_seed(seed):
    """seed=-1 → 每次请求随机抽一个。server/C++ 侧 seed 一律按无符号整数解析
    （多数族 u32，seed_vc/stable_audio u64），负数会直接报错，所以 -1 只能在
    客户端消化；随机范围取 u32 全集，对所有族安全。返回 (seed, 消息后缀)——
    后缀把实际用的 seed 回显在结果里，方便复现。"""
    s = int(seed)
    if s != -1:
        return s, ""
    s = random.randrange(0, 2 ** 32)
    return s, f"🎲 seed={s}"


def do_tts(model, text, language, uploaded_voice, builtin_voice,
           reference_text, seed, max_tokens, adv_values, adv_options,
           progress=gr.Progress()):
    try:
        if not (text or "").strip():
            raise gr.Error(t("err.no_text"))

        entry = catalog_by_id(model)
        prof = profile_for(entry) if entry else DEFAULT_PROFILE

        if prof.get("wrap_speaker_script"):     # e.g. VibeVoice needs Speaker N: lines
            text = _as_speaker_script(text)
            text = _vibevoice_punctuate_script(text)

        # 超短文本拦截（见 _VIBEVOICE_MIN_EST_TOKENS）——放在模型加载之前，
        # 免得为一个注定被拒的请求重启 server。
        is_vibevoice = bool(entry) and entry.get("family") == "vibevoice"
        if is_vibevoice and _vibevoice_text_max_tokens(text) < _VIBEVOICE_MIN_EST_TOKENS:
            raise gr.Error(t("err.vibevoice_text_too_short"))

        ensure_model_loaded(model, TTS_TASKS)

        # Model-specific knobs travel in a nested "options" object; the server merges
        # every key into the request options and each model reads what it understands.
        options = _merged_options(prof, adv_values, adv_options)

        voice_path = None
        if uploaded_voice:                      # gradio gives an absolute temp path
            voice_path = uploaded_voice
        elif builtin_voice and builtin_voice != "(none)":
            voice_path = os.path.join(PROMPTS_DIR, builtin_voice)
        has_voice_samples = "voice_samples" in options or "vibevoice.voice_samples" in options
        if is_vibevoice and voice_path and not has_voice_samples:
            options["voice_samples"] = _ensure_wav(voice_path)
            voice_path = None
        # voice_samples (multi-speaker) can't be combined with a single voice_ref.
        if has_voice_samples and voice_path:
            voice_path = None

        seed, seed_note = _resolve_seed(seed)
        payload = {
            "model": model,
            "language": resolve_language(prof, language),
            "seed": seed,
            "max_tokens": int(max_tokens),
        }
        auto_vibevoice_max_tokens = (
            is_vibevoice and int(max_tokens) == 1200
        )
        if voice_path:
            payload["voice_ref"] = _ensure_wav(voice_path)
        if (reference_text or "").strip():
            payload["reference_text"] = reference_text
        if options:
            payload["options"] = options

        # Long text goes out as several bounded requests (concatenated below), so a
        # whole chapter neither hits the per-request timeout nor runs blind.
        chunks = _split_tts_chunks(text, prof.get("chunk_chars", 1000))
        if is_vibevoice:
            chunks = _merge_short_vibevoice_tail(chunks)
        _ui_log(t("log.tts_start", model=model, chunks=len(chunks),
                  chars=sum(len(c) for c in chunks)))
        t_start = time.time()
        blobs = []
        for i, chunk in enumerate(chunks):
            if len(chunks) > 1:
                progress((i, len(chunks)), desc=t("progress.tts_chunk", i=i + 1, n=len(chunks)))
            payload["input"] = chunk
            if auto_vibevoice_max_tokens:
                payload["max_tokens"] = _vibevoice_text_max_tokens(chunk, int(max_tokens))
            t_chunk = time.time()
            try:
                r = requests.post(f"{SERVER}/v1/audio/speech", json=payload, timeout=900)
            except requests.RequestException as e:
                _ui_log(t("log.tts_failed_unreachable", i=i + 1, n=len(chunks)))
                raise gr.Error(t("err.server_unreachable", server=SERVER, error=e, load=t("btn.load")))
            if r.status_code != 200:
                _ui_log(t("log.tts_failed_status", i=i + 1, n=len(chunks), status=r.status_code))
                raise server_error(entry, r.status_code, r.text)
            blobs.append(r.content)
            _ui_log(t("log.tts_chunk_done", i=i + 1, n=len(chunks), chars=len(chunk),
                      secs=f"{time.time() - t_chunk:.1f}"))

        out = os.path.join(OUTPUT_DIR, f"audiocpp_tts_{int(time.time()*1000)}.wav")
        if len(blobs) == 1:
            with open(out, "wb") as f:
                f.write(blobs[0])
        else:
            _concat_wavs(blobs, out)
        elapsed = time.time() - t_start
        _ui_log(t("log.tts_done", out=out, secs=f"{elapsed:.1f}"))
        parts_note = t("label.n_parts", n=len(blobs)) if len(blobs) > 1 else ""
        return out, t("msg.generate_done", parts=parts_note, secs=f"{elapsed:.1f}",
                      seed=seed_note)
    except gr.Error as e:
        return None, _msg_from_error(e)
    except Exception as e:
        return None, t("msg.generate_failed", error=e)


def _asr_transcribe_wav(model, entry, prof, wav_path, extras, tag="ASR"):
    """转写一个 WAV 文件（调用方已加载好 ASR 模型）：超过单次上限（见 qwen3_asr
    profile：8G 卡显存实测取 60s）时在静音处切段逐段转写再拼接；短音频/非 PCM
    WAV（测不出时长）走单请求。extras 是可选的 language/context 请求字段。
    返回 (text, dur, 段数)。"""
    dur = _audio_duration_seconds(wav_path)
    dur_note = f"{dur:.1f}s" if dur is not None else t("label.unknown")
    max_s = prof.get("max_input_seconds")
    chunks = [wav_path]
    if max_s and dur is not None and dur > max_s:
        chunks = [p for p, _ in
                  _split_wav_chunks(wav_path, max_s, pad_to_max=False)]
        if len(chunks) > 1:
            _ui_log(t("log.asr_chunking", tag=tag, dur=dur_note, n=len(chunks),
                      max_s=f"{max_s:.0f}"))
    texts = []
    for i, chunk in enumerate(chunks):
        t_chunk = time.time()
        payload = {"model": model, "audio": chunk, **extras}
        try:
            r = requests.post(f"{SERVER}/v1/audio/transcriptions", json=payload, timeout=900)
        except requests.RequestException as e:
            _ui_log(t("log.tag_failed_unreachable", tag=tag))
            raise gr.Error(t("err.server_unreachable", server=SERVER, error=e, load=t("btn.load")))
        if r.status_code != 200:
            seg_note = t("log.seg_note", i=i + 1, n=len(chunks)) if len(chunks) > 1 else ""
            _ui_log(t("log.tag_failed_status", tag=tag, status=r.status_code,
                      dur=dur_note, seg=seg_note))
            extra = t("err.audio_duration_note", secs=f"{dur:.1f}") if dur is not None else None
            raise server_error(entry, r.status_code, r.text, extra=extra)
        try:
            data = r.json()
            texts.append((data.get("text") or str(data)).strip())
        except Exception:
            texts.append(r.text)
        if len(chunks) > 1:
            _ui_log(t("log.tag_chunk_done", tag=tag, i=i + 1, n=len(chunks),
                      secs=f"{time.time() - t_chunk:.1f}"))
    text = texts[0] if len(chunks) == 1 else "\n".join(x for x in texts if x)
    return text, dur, len(chunks)


def do_asr(model, audio_path, language="", context="", dialogue=False):
    try:
        if not audio_path:
            raise gr.Error(t("err.no_audio"))
        audio_path = _ensure_wav(audio_path)
        # 可选转写参数：留空不发，请求体和原来完全一致（qwen3_asr 从 text_input
        # 读 context/language，其它族忽略；server 端 build_openai_transcription_request
        # 只在字段存在时才设置 text_input）。
        extras = {}
        if (language or "").strip():
            extras["language"] = language.strip()
        if (context or "").strip():
            extras["context"] = context.strip()
        if dialogue:
            return _asr_dialogue(model, audio_path, extras)
        ensure_model_loaded(model, ASR_TASKS)
        entry = catalog_by_id(model)
        prof = profile_for(entry) if entry else DEFAULT_PROFILE
        extras_note = "".join(f", {k}={v[:20]}" for k, v in extras.items())
        _ui_log(t("log.asr_start", model=model, extras=extras_note))
        t_start = time.time()
        text, dur, n = _asr_transcribe_wav(model, entry, prof, audio_path, extras)
        elapsed = time.time() - t_start
        dur_note = f"{dur:.1f}s" if dur is not None else t("label.unknown")
        parts_note = t("label.n_chunks", n=n) if n > 1 else ""
        _ui_log(t("log.asr_done", dur=dur_note, parts=parts_note, secs=f"{elapsed:.1f}"))
        return text, t("msg.asr_done", dur=dur_note, parts=parts_note, secs=f"{elapsed:.1f}")
    except gr.Error as e:
        return "", _msg_from_error(e)
    except Exception as e:
        return "", t("msg.transcribe_failed", error=e)


# 对话模式：同一说话人相邻发言段合并的最大间隔 / 每段前后补的余量（防止
# Sortformer 边界切掉字头字尾）/ 短于该值的发言段丢弃（多为口头禅、气口）。
DIALOGUE_MERGE_GAP_S = 1.0
DIALOGUE_PAD_S = 0.25
DIALOGUE_MIN_SEG_S = 0.3

# Sortformer CUDA 下是固定图容量：session_len_sec 定容量（默认 20s），上限来自
# tf_encoder.max_source_positions=1500 @ 12.5 编码帧/秒 = 120s。
SORTFORMER_MAX_SEC = 120


def _diar_session_options(dur):
    """说话人分离的动态 session_len_sec：≤20s 用默认容量（不重载）；更长的按
    30s 步进向上取整重载（避免每个文件都重启 server）；超过 120s 上限报错。"""
    if dur is None or dur <= 20:
        return None
    if dur > SORTFORMER_MAX_SEC:
        raise gr.Error(
            t("err.diar_too_long", max_sec=SORTFORMER_MAX_SEC))
    return {"session_len_sec": str(min(SORTFORMER_MAX_SEC,
                                       ((int(dur) // 30) + 1) * 30))}


def _fmt_mmss(sec):
    return f"{int(sec) // 60:02d}:{int(sec) % 60:02d}"


def _first_installed_model(task):
    for m in catalog_models():
        if m.get("task") == task and m.get("installed"):
            return m
    return None


def _asr_dialogue(model, wav_path, extras):
    """对话模式：Sortformer 说话人分离 → 按说话人合并/切段 → 逐段 ASR →
    带说话人标签和时间戳的对话稿。server 一次只驻留一个模型，所以先换载 diar
    再换回 ASR（每次自动换载 ~5s）。"""
    diar = _first_installed_model("diar")
    if diar is None:
        raise gr.Error(t("err.dialogue_needs_diar", tab=t("tab.analyze")))
    entry = catalog_by_id(model)
    prof = profile_for(entry) if entry else DEFAULT_PROFILE

    wav16 = _to_16k_mono_wav(wav_path)
    try:
        with wave.open(wav16, "rb") as w:
            sr = w.getframerate()
            raw = w.readframes(w.getnframes())
    except Exception as e:
        raise gr.Error(t("err.dialogue_needs_pcm", error=e))
    samples = np.frombuffer(raw, dtype=np.int16)
    total_dur = len(samples) / float(sr)

    t_start = time.time()
    _ui_log(t("log.dialogue_diar_first", diar=diar["id"], dur=f"{total_dur:.1f}"))
    ensure_model_loaded(diar["id"], ("diar",),
                        session_options=_diar_session_options(total_dur))
    data = _run_task(diar, diar["id"], {"audio": wav16}, timeout=900,
                     log_label=t("task.diarization"))
    turns = data.get("speaker_turns") or []
    if not turns:
        raise gr.Error(t("err.no_speaker_turns"))

    # 合并同一说话人的相邻发言段（间隔 ≤1s 且合并后不超过 ASR 单次上限），
    # 减少请求数并给 ASR 更完整的上下文。
    max_len = (prof.get("max_input_seconds") or 60) * sr
    merged = []                                     # [说话人, start, end] (样本数)
    for turn in sorted(turns, key=lambda x: x["start_sample"]):
        spk, s, e = (str(turn.get("speaker_id", "?")),
                     turn["start_sample"], turn["end_sample"])
        if (merged and merged[-1][0] == spk
                and s - merged[-1][2] <= DIALOGUE_MERGE_GAP_S * sr
                and e - merged[-1][1] <= max_len):
            merged[-1][2] = max(merged[-1][2], e)
        else:
            merged.append([spk, s, e])
    merged = [m for m in merged if m[2] - m[1] >= DIALOGUE_MIN_SEG_S * sr]
    if not merged:
        raise gr.Error(t("err.turns_too_short"))
    _ui_log(t("log.diar_done", turns=len(turns), merged=len(merged), model=model))

    ensure_model_loaded(model, ASR_TASKS)
    pad = int(DIALOGUE_PAD_S * sr)
    speakers, lines = [], []
    for idx, (spk, s, e) in enumerate(merged, 1):
        a, b = max(0, s - pad), min(len(samples), e + pad)
        fd, seg_path = tempfile.mkstemp(prefix=f"audiocpp_dlg{idx}_", suffix=".wav")
        os.close(fd)
        with wave.open(seg_path, "wb") as ww:
            ww.setnchannels(1)
            ww.setsampwidth(2)
            ww.setframerate(sr)
            ww.writeframes(samples[a:b].tobytes())
        text, _, _ = _asr_transcribe_wav(model, entry, prof, seg_path, extras,
                                         tag=t("label.dialogue_seg", i=idx, n=len(merged)))
        if spk not in speakers:
            speakers.append(spk)
        label = t("label.speaker_n", n=speakers.index(spk) + 1)
        if text:
            lines.append(f"[{_fmt_mmss(s / sr)}-{_fmt_mmss(e / sr)}] {label}: {text}")
        _ui_log(t("log.dialogue_seg_done", i=idx, n=len(merged), label=label,
                  secs=f"{(e - s) / sr:.1f}"))

    elapsed = time.time() - t_start
    _ui_log(t("log.dialogue_done", turns=len(merged), speakers=len(speakers),
              secs=f"{elapsed:.1f}"))
    if not lines:
        return "", t("msg.dialogue_no_text")
    return ("\n".join(lines),
            t("msg.dialogue_done", dur=f"{total_dur:.1f}", turns=len(merged),
              speakers=len(speakers), secs=f"{elapsed:.1f}"))


def do_music_gen(model, text, lyrics, source_audio, duration, seed,
                 adv_values, adv_options):
    """Music/SFX generation via the generic /v1/tasks/run route. The request
    object uses the CLI request-JSON fields (text/lyrics/duration_seconds/
    task_route/audio + an options map); the response carries base64 WAV."""
    try:
        if not (text or "").strip():
            raise gr.Error(t("err.no_prompt"))
        ensure_model_loaded(model, GEN_TASKS)
        entry = catalog_by_id(model)
        prof = profile_for(entry) if entry else DEFAULT_PROFILE
        options = _merged_options(prof, adv_values, adv_options)

        seed, seed_note = _resolve_seed(seed)
        req = {"text": text, "seed": seed}
        # task_route is a top-level request field (not a model option); the
        # generated controls funnel everything through `options`, so lift it out.
        route = options.pop("task_route", None)
        if route:
            req["task_route"] = route
        if (lyrics or "").strip():
            req["lyrics"] = lyrics
        if duration is not None and float(duration) != 0:
            req["duration_seconds"] = float(duration)
        if source_audio:
            req["audio"] = _ensure_wav(source_audio)
        if options:
            req["options"] = options

        dur_note = req.get("duration_seconds", t("label.auto"))
        _ui_log(t("log.gen_start", model=model, dur=dur_note))
        t_start = time.time()
        data = _run_task(entry, model, req, timeout=1800, log_label=t("task.music_gen"))
        b64 = data.get("audio")
        if not b64 and data.get("named_audio_outputs"):
            b64 = data["named_audio_outputs"][0].get("audio")
        if not b64:
            raise gr.Error(t("err.no_audio_in_response_gen"))
        out = os.path.join(OUTPUT_DIR, f"audiocpp_gen_{int(time.time()*1000)}.wav")
        with open(out, "wb") as f:
            f.write(base64.b64decode(b64))
        elapsed = time.time() - t_start
        _ui_log(t("log.gen_done", out=out, secs=f"{elapsed:.1f}"))
        return out, t("msg.generate_done", parts="", secs=f"{elapsed:.1f}", seed=seed_note)
    except gr.Error as e:
        return None, _msg_from_error(e)
    except Exception as e:
        return None, t("msg.generate_failed", error=e)


# analyze 返回的字段 -> ACE-Step 高级参数（model_params.json 里的 name）
_ANALYZE_FILL_MAP = (("caption", "source_caption"), ("lyrics", "source_lyrics"),
                     ("bpm", "bpm"), ("keyscale", "keyscale"),
                     ("timesignature", "timesignature"))


def do_music_analyze(model, source_audio, seed, adv_values):
    """『🔍 分析源音频』（仅 ACE-Step）：task_route=analyze 把源音频编码成语义 code，
    再用 5Hz LM 反推 caption/歌词/BPM/调性/拍号，回填到高级参数
    （写进 state 并通过 prefill state 触发控件重渲染，让填进去的值可见可改）。
    返回 (adv_state, prefill, message)。"""
    adv_values = dict(adv_values or {})
    try:
        if not source_audio:
            raise gr.Error(t("err.no_source_audio"))
        entry = catalog_by_id(model)
        if not entry or entry.get("family") != "ace_step":
            raise gr.Error(t("err.analysis_ace_only"))
        ensure_model_loaded(model, GEN_TASKS)

        # 分析要可复现：seed=-1（随机）时固定为 1234，不跟生成共享随机性；
        # 用户显式填的固定 seed 仍然生效（可换 seed 重抽歌词转写）。
        analyze_seed = 1234 if int(seed if seed is not None else -1) == -1 else int(seed)
        req = {"text": "analyze", "task_route": "analyze",
               "audio": _ensure_wav(source_audio), "seed": analyze_seed}
        dur = _audio_duration_seconds(req["audio"])
        dur_note = f"{dur:.1f}s" if dur is not None else t("label.unknown")
        _ui_log(t("log.src_analysis_start", model=model, dur=dur_note))
        t_start = time.time()
        data = _run_task(entry, model, req, timeout=1800, log_label=t("task.source_analysis"))
        raw = data.get("text") or ""
        try:
            info = json.loads(raw)
        except Exception:
            raise gr.Error(t("err.analysis_not_json", raw=raw[:200]))
        elapsed = time.time() - t_start

        for src, dst in _ANALYZE_FILL_MAP:
            v = info.get(src)
            if v is None or v == "" or v == 0:
                continue
            adv_values[dst] = v

        lines = [t("msg.src_analysis_done", dur=dur_note, secs=f"{elapsed:.1f}")]
        for key, label in (("caption", t("field.caption")), ("bpm", "BPM"),
                           ("keyscale", t("field.keyscale")),
                           ("timesignature", t("field.timesignature")),
                           ("language", t("field.language")),
                           ("genres", t("field.genres")), ("duration", t("field.duration"))):
            v = info.get(key)
            if v not in (None, "", 0):
                lines.append(f"- **{label}**：{v}")
        if info.get("lyrics"):
            lines.append("- **" + t("field.lyrics") + f"**:\n```\n{info['lyrics']}\n```")
        _ui_log(t("log.src_analysis_done", secs=f"{elapsed:.1f}"))
        return adv_values, dict(adv_values), "\n".join(lines)
    except gr.Error as e:
        return adv_values, gr.skip(), _msg_from_error(e)
    except Exception as e:
        return adv_values, gr.skip(), t("msg.analyze_failed", error=e)


def do_vc(model, source_audio, target_upload, builtin_voice, seed,
          adv_values, adv_options, progress=gr.Progress()):
    """声音/歌声转换（vc/svc/s2s），走通用 /v1/tasks/run 路由：`audio` 是源音频，
    `voice_ref` 是目标音色。seed_vc/miocodec 直接用这两个字段；vevo2 也接受它们
    （audio_input/voice speaker 是 source_audio/target_voice 选项的回退），
    风格转换类 route 的额外字段（style_ref 等）由“其它参数(JSON)”兜底。
    profile 带 vc_chunk_seconds 的族（vevo2 FM 图按整段建，8G 卡长音频必炸）
    超限时按低能量点分段逐段转换，同一 voice_ref 保证各段音色一致，最后拼接。"""
    try:
        if not source_audio:
            raise gr.Error(t("err.no_source_to_convert"))
        ensure_model_loaded(model, VC_TASKS)
        entry = catalog_by_id(model)
        prof = profile_for(entry) if entry else DEFAULT_PROFILE
        options = _merged_options(prof, adv_values, adv_options)

        source_audio = _ensure_wav(source_audio)
        # 同一 seed 用于所有分段，保证各段结果一致可复现。
        seed, seed_note = _resolve_seed(seed)
        req = {"seed": seed}
        voice_path = target_upload or (
            os.path.join(PROMPTS_DIR, builtin_voice)
            if builtin_voice and builtin_voice != "(none)" else None)
        if voice_path:
            ref_wav = _ensure_wav(voice_path)
            ref_cap = prof.get("vc_ref_max_seconds")
            if ref_cap:                     # 参考音色截短，别让 prompt 段吃满显存
                ref_wav = _trim_wav_seconds(ref_wav, ref_cap)
            req["voice_ref"] = ref_wav
        if options:
            req["options"] = options

        # vevo2 的 FM 图一次建图，序列长度 = 参考音色(prompt) + 每段源(target)。按
        # 显存预算反推每段源时长（预算 − 参考时长），令 cond 稳定落在 8G 内；没有
        # 显存预算/参考时的族仍按 vc_chunk_seconds 上限或整段发送。
        chunk_cap = prof.get("vc_chunk_seconds")
        ref_sec = _audio_duration_seconds(req.get("voice_ref")) or 0.0
        if chunk_cap and prof.get("vc_fm_budget_seconds"):
            min_chunk = prof.get("vc_min_chunk_seconds", 6)
            budget = prof["vc_fm_budget_seconds"]
            chunk_cap = int(max(min_chunk, min(chunk_cap, round(budget - ref_sec))))
        pieces = (_split_wav_chunks(source_audio, chunk_cap)
                  if chunk_cap else [(source_audio, 1.0)])
        dur = _audio_duration_seconds(source_audio)
        dur_note = f"{dur:.1f}s" if dur is not None else t("label.unknown")
        seg_note = (t("log.vc_seg_note", ref=f"{ref_sec:.0f}", n=len(pieces), cap=chunk_cap)
                    if len(pieces) > 1 else "")
        _ui_log(t("log.vc_start", model=model, dur=dur_note, seg=seg_note))
        t_start = time.time()
        blobs, ratios = [], []
        for i, (piece, ratio) in enumerate(pieces):
            if len(pieces) > 1:
                progress((i, len(pieces)), desc=t("progress.vc_chunk", i=i + 1, n=len(pieces)))
            req["audio"] = piece
            t_seg = time.time()
            data = _run_task(entry, model, req, timeout=1800, log_label=t("task.voice_conversion"))
            b64 = data.get("audio")
            if not b64 and data.get("named_audio_outputs"):
                b64 = data["named_audio_outputs"][0].get("audio")
            if not b64:
                raise gr.Error(t("err.no_audio_in_response"))
            blobs.append(base64.b64decode(b64))
            ratios.append(ratio)
            if len(pieces) > 1:
                _ui_log(t("log.vc_chunk_done", i=i + 1, n=len(pieces),
                          secs=f"{time.time() - t_seg:.1f}"))
        out = os.path.join(OUTPUT_DIR, f"audiocpp_vc_{int(time.time()*1000)}.wav")
        if len(blobs) == 1 and ratios[0] >= 1.0:
            with open(out, "wb") as f:
                f.write(blobs[0])
        else:
            _concat_wavs(blobs, out, keep_ratios=ratios)
        elapsed = time.time() - t_start
        _ui_log(t("log.vc_done", out=out, secs=f"{elapsed:.1f}"))
        parts_note = t("label.n_parts_joined", n=len(blobs)) if len(blobs) > 1 else ""
        return out, t("msg.convert_done", parts=parts_note, secs=f"{elapsed:.1f}", seed=seed_note)
    except gr.Error as e:
        return None, _msg_from_error(e)
    except Exception as e:
        return None, t("msg.convert_failed", error=e)


# Stem id -> localized label; ids not listed here are shown verbatim.
STEM_LABELS = {k: t("stem." + k) for k in
               ("vocals", "drums", "bass", "other", "instrumental",
                "accompaniment", "audio")}


def _rebuild_localized_tables():
    """Recompute the module-level tables that call t() at import time.

    set_language() calls this. Anything built once from t() would otherwise keep
    the language that was active at import, so a switch would leave model hints,
    stem names and the ASR language list in the old language.
    """
    global STEM_LABELS
    STEM_LABELS = {k: t("stem." + k) for k in STEM_LABELS}
    for family, profile in MODEL_PROFILES.items():
        if "input_hint" in profile:
            profile["input_hint"] = t("hint." + family)


MAX_SEP_STEMS = 4
# 分离族（htdemucs / mel-band-roformer）的 prepare() 硬校验 44.1kHz（模型配置
# sample_rate），不自己重采样；其它采样率的输入在 webui 侧先转到 44.1k。
SEP_SR = 44100


def do_sep(model, audio_path):
    """音源分离：响应里的 named_audio_outputs 每轨落盘成一个 wav，
    前 MAX_SEP_STEMS 轨直接放进播放器，全部轨放进文件下载列表。"""
    empty = [gr.update(value=None, visible=False) for _ in range(MAX_SEP_STEMS)]
    try:
        if not audio_path:
            raise gr.Error(t("err.no_audio_to_separate"))
        audio_path = _ensure_wav(audio_path, target_sr=SEP_SR)
        ensure_model_loaded(model, SEP_TASKS)
        entry = catalog_by_id(model)
        dur = _audio_duration_seconds(audio_path)
        dur_note = f"{dur:.1f}s" if dur is not None else t("label.unknown")
        _ui_log(t("log.sep_start", model=model, dur=dur_note))
        t_start = time.time()
        data = _run_task(entry, model, {"audio": audio_path},
                         timeout=1800, log_label=t("task.separation"))
        stems = data.get("named_audio_outputs") or []
        if not stems and data.get("audio"):
            stems = [{"id": "audio", "audio": data["audio"]}]
        if not stems:
            raise gr.Error(t("err.no_stems_in_response"))
        ts = int(time.time() * 1000)
        paths = []
        for stem in stems:
            sid = stem.get("id") or f"stem{len(paths)}"
            safe = re.sub(r"[^A-Za-z0-9_.-]", "_", sid)
            p = os.path.join(OUTPUT_DIR, f"audiocpp_sep_{ts}_{safe}.wav")
            with open(p, "wb") as f:
                f.write(base64.b64decode(stem["audio"]))
            paths.append((sid, p))
        updates = []
        for i in range(MAX_SEP_STEMS):
            if i < len(paths):
                sid, p = paths[i]
                zh = STEM_LABELS.get(sid)
                updates.append(gr.update(
                    value=p, visible=True, label=f"{zh}（{sid}）" if zh else sid))
            else:
                updates.append(gr.update(value=None, visible=False))
        elapsed = time.time() - t_start
        _ui_log(t("log.sep_done", n=len(paths), secs=f"{elapsed:.1f}"))
        note = ("" if len(paths) <= MAX_SEP_STEMS
                else t("label.extra_stems", n=len(paths) - MAX_SEP_STEMS))
        return (*updates, [p for _, p in paths],
                t("msg.sep_done", n=len(paths), note=note, secs=f"{elapsed:.1f}"))
    except gr.Error as e:
        return (*empty, None, _msg_from_error(e))
    except Exception as e:
        return (*empty, None, t("msg.separate_failed", error=e))


# VAD/diar/align 输入统一转成 16 kHz 单声道后再发（见 _to_16k_mono_wav），
# 所以响应里的 start_sample/end_sample 一律按 16000 换算成秒。
SR_ANALYZE = 16000


def _fmt_ts(samples):
    return f"{samples / SR_ANALYZE:.2f}s"


def do_analyze(model, audio_path, transcript, language):
    """音频分析（vad/diar/align）：格式化 segments / speaker_turns / words 为
    可读文本，原始 JSON 落盘供下载。"""
    try:
        if not audio_path:
            raise gr.Error(t("err.no_audio"))
        entry = catalog_by_id(model)
        task = entry.get("task") if entry else ""
        req = {"audio": _to_16k_mono_wav(_ensure_wav(audio_path))}
        if task == "align":
            if not (transcript or "").strip():
                raise gr.Error(t("err.align_needs_transcript"))
            req["text"] = transcript.strip()
            if (language or "").strip():
                req["language"] = language.strip()
        dur = _audio_duration_seconds(req["audio"])
        # diar（Sortformer）是固定图容量，>20s 的音频按时长动态重载。
        ensure_model_loaded(model, ANALYZE_TASKS,
                            session_options=(_diar_session_options(dur)
                                             if task == "diar" else None))
        dur_note = f"{dur:.1f}s" if dur is not None else t("label.unknown")
        _ui_log(t("log.analyze_start", model=model, task=task, dur=dur_note))
        t_start = time.time()
        data = _run_task(entry, model, req, timeout=900, log_label=t("task.audio_analysis"))

        lines = []
        if data.get("segments"):
            segs = data["segments"]
            lines.append(t("analyze.n_segments", n=len(segs)))
            speech = 0
            for i, s in enumerate(segs, 1):
                lines.append(f"{i:3d}. {_fmt_ts(s['start_sample'])} → {_fmt_ts(s['end_sample'])}"
                             + "\u3000" + t("analyze.confidence", value=f"{s.get('confidence', 0):.2f}"))
                speech += s["end_sample"] - s["start_sample"]
            lines.append(t("analyze.total_speech", secs=f"{speech / SR_ANALYZE:.1f}"))
        if data.get("speaker_turns"):
            turns = data["speaker_turns"]
            spk = sorted({turn.get("speaker_id", "?") for turn in turns})
            lines.append(t("analyze.n_turns", n=len(turns), speakers=len(spk),
                           ids=", ".join(spk)))
            for i, turn in enumerate(turns, 1):
                lines.append(f"{i:3d}. {_fmt_ts(turn['start_sample'])} → {_fmt_ts(turn['end_sample'])}"
                             + "\u3000" + str(turn.get("speaker_id", "?")) + "\u3000"
                             + t("analyze.confidence", value=f"{turn.get('confidence', 0):.2f}"))
        if data.get("words"):
            lines.append(t("analyze.n_words", n=len(data["words"])))
            for w in data["words"]:
                lines.append(f"{_fmt_ts(w['start_sample'])} → {_fmt_ts(w['end_sample'])}"
                             + "\u3000" + str(w.get("word", "")))
        if data.get("text"):
            lines.append(t("analyze.text_output", text=data["text"]))
        if not lines:
            lines.append(t("analyze.no_results"))

        json_path = os.path.join(OUTPUT_DIR, f"audiocpp_analyze_{int(time.time()*1000)}.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        elapsed = time.time() - t_start
        _ui_log(t("log.analyze_done", secs=f"{elapsed:.1f}"))
        return ("\n".join(lines), json_path,
                t("msg.analyze_done", dur=dur_note, secs=f"{elapsed:.1f}"))
    except gr.Error as e:
        return "", None, _msg_from_error(e)
    except Exception as e:
        return "", None, t("msg.analyze_failed", error=e)


def _align_fields_visibility(model_id):
    """音频分析页：只有 align 任务的模型才显示『对齐文本/语言』输入。"""
    entry = catalog_by_id(model_id) if model_id else None
    show = bool(entry) and entry.get("task") == "align"
    return gr.update(visible=show), gr.update(visible=show)


def do_vdes(model, text, instruct, seed, max_tokens, adv_values, adv_options):
    """声音设计（vdes）：文字 + 音色描述走 /v1/audio/speech（instructions 字段
    映射到模型的 instruct 选项），响应是 WAV 音频。"""
    try:
        if not (text or "").strip():
            raise gr.Error(t("err.no_text"))
        if not (instruct or "").strip():
            raise gr.Error(t("err.vdes_needs_description"))
        ensure_model_loaded(model, VDES_TASKS)
        entry = catalog_by_id(model)
        prof = profile_for(entry) if entry else DEFAULT_PROFILE
        options = _merged_options(prof, adv_values, adv_options)

        seed, seed_note = _resolve_seed(seed)
        payload = {"model": model, "input": text, "instructions": instruct,
                   "seed": seed, "max_tokens": int(max_tokens)}
        if options:
            payload["options"] = options
        _ui_log(t("log.vdes_start", model=model, chars=len(text)))
        t_start = time.time()
        try:
            r = requests.post(f"{SERVER}/v1/audio/speech", json=payload, timeout=900)
        except requests.RequestException as e:
            _ui_log(t("log.vdes_failed_unreachable"))
            raise gr.Error(t("err.server_unreachable", server=SERVER, error=e, load=t("btn.load")))
        if r.status_code != 200:
            _ui_log(t("log.vdes_failed_status", status=r.status_code))
            raise server_error(entry, r.status_code, r.text)
        out = os.path.join(OUTPUT_DIR, f"audiocpp_vdes_{int(time.time()*1000)}.wav")
        with open(out, "wb") as f:
            f.write(r.content)
        elapsed = time.time() - t_start
        _ui_log(t("log.vdes_done", out=out, secs=f"{elapsed:.1f}"))
        return out, t("msg.generate_done", parts="", secs=f"{elapsed:.1f}", seed=seed_note)
    except gr.Error as e:
        return None, _msg_from_error(e)
    except Exception as e:
        return None, t("msg.generate_failed", error=e)


def _make_load_handler(tasks):
    """『📥 加载模型』按钮的处理器工厂：按标签页各自的 task 集合校验并加载。"""
    def _load(model):
        try:
            s = ensure_model_loaded(model, tasks)
        except gr.Error as e:
            s = _msg_from_error(e)
        return s, server_status()
    return _load


# 标签页顺序（与 refresh() 的输出、_refresh_outputs 列表一一对应）：
# TTS、ASR、音乐生成、声音转换、音源分离、音频分析、声音设计。
TAB_SPECS = [TTS_TASKS, ASR_TASKS, GEN_TASKS, VC_TASKS,
             SEP_TASKS, ANALYZE_TASKS, VDES_TASKS]


# --- live language switching ------------------------------------------------
# Gradio builds the Blocks tree once, baking each label in at construction, and it
# offers no way to rebuild it in another language. So the picker re-labels every
# component in place: this registry remembers which locale key produced which
# property, and _relabel_updates() turns it back into one gr.update per component.
#
# The registry is derived rather than declared. Every localized string already goes
# through t(), so a component's English text is enough to identify its key by
# reverse lookup — which keeps ~120 call sites free of registration boilerplate.
# Where several keys share one English string, reverse lookup cannot tell them
# apart, so _english_key_index() checks whether the translations agree and drops the
# ones that don't: such a component keeps its current label instead of risking the
# wrong one.

# Components whose `value` is display text rather than user data. Everything else
# (Textbox, Dropdown, ...) holds a value the user or the wire owns: re-labelling it
# would silently corrupt a request — e.g. a language Dropdown whose value is the
# literal "Chinese" the server expects.
_TEXT_VALUE_TYPES = (gr.Markdown, gr.Button, gr.HTML)
_LABEL_PROPS = ("label", "placeholder", "info")

_i18n_registry = []          # [(component, prop, key, format_args)]


def i18n_register(comp, prop, key, **fmt):
    """Register a property reverse lookup cannot find, and return `comp`.

    Needed for strings built with format arguments: the component shows
    "Model list (task=tts)" while the locale holds "Model list (task={task})", so
    matching on text fails. Interpolated labels must say so here.
    """
    _i18n_registry.append((comp, prop, key, fmt))
    return comp


def _english_key_index():
    """English string -> locale key, for identifying components by their text.

    Keys sharing one English string are ambiguous by reverse lookup. They are only
    a problem if some locale translates them differently, so check every installed
    locale and drop the ones that disagree rather than guess.
    """
    by_text = {}
    for key, text in _FALLBACK_STRINGS.items():
        if isinstance(text, str) and text:
            by_text.setdefault(text, []).append(key)

    others = [_load_locale(code) for code in available_langs() if code != DEFAULT_LANG]
    index = {}
    for text, keys in by_text.items():
        if len(keys) > 1:
            ambiguous = any(len({loc.get(k, text) for k in keys}) > 1 for loc in others)
            if ambiguous:
                logging.warning("keys %s share the English text %r but differ when "
                                "translated; they will not switch language", keys, text)
                continue
        index[text] = keys[0]
    return index


def build_i18n_registry(blocks):
    """Record which locale key produced each localized property of each component."""
    index = _english_key_index()
    registry = []
    for comp in blocks:
        for prop in _LABEL_PROPS:
            text = getattr(comp, prop, None)
            if isinstance(text, str) and text in index:
                registry.append((comp, prop, index[text], {}))
        if isinstance(comp, _TEXT_VALUE_TYPES):
            text = getattr(comp, "value", None)
            if isinstance(text, str) and text in index:
                registry.append((comp, "value", index[text], {}))
    return registry


def _relabel_updates():
    """(components, updates) re-localizing every registered property."""
    grouped = {}
    for comp, prop, key, fmt in _i18n_registry:
        grouped.setdefault(comp, {})[prop] = t(key, **fmt)
    comps = list(grouped)
    return comps, [gr.update(**props) for props in grouped.values()]


# A locale may name itself via "lang.self"; otherwise the code is all we can show.
# Deliberately not localized: someone stuck in a language they cannot read needs to
# recognize their own language in this list.
_LANG_SELF_NAMES = {"en": "English", "zh": "中文"}


def language_choices():
    """(display name, code) for every locales/<code>.json present."""
    return [(_LANG_SELF_NAMES.get(code, code), code) for code in available_langs()]


def refresh():
    """重读 catalog / 参数配置，刷新每个标签页的模型下拉和提示。
    返回顺序：各页下拉更新（按 TAB_SPECS 顺序）、状态行、各页提示。"""
    global CATALOG, MODEL_PARAMS
    CATALOG = _load_catalog()
    MODEL_PARAMS = _load_model_params()
    dropdowns, hints = [], []
    for tasks in TAB_SPECS:
        choices = choices_for_tasks(tasks)
        value = choices[0][1] if choices else None
        dropdowns.append(gr.update(choices=choices, value=value))
        hints.append(model_hint_for(value))
    return (*dropdowns, server_status(), *hints)


atexit.register(_stop_server)


CUSTOM_CSS = """

.audio-default { border: none !important; box-shadow: none !important; }

.mm-btn-row { gap: 10px !important; }
.mm-btn-row button { border-radius: var(--button-large-radius, var(--radius-lg)) !important; }

/* 长音频的波形出现横向滚动条时，WaveSurfer 的滚动层（58px 内容 + 滚动条）会
   溢出 Gradio 固定 58px 的 .waveform-container / #waveform，盖住下方的
   0:00/总时长标签。放开这两层的高度让标签随内容下移；短音频（无滚动条）时
   min-height 保证布局与原来一致。 */
.waveform-container, #waveform { height: auto !important; min-height: 58px; }

.hint-small { opacity: 0.7; font-size: 0.85em; margin-top: 2px; }

/* 内置参考音色行：Row 本身透明，会在按钮一列露出 gr.Group 的灰色面板底，
   和 secondary 按钮的灰底连成一片。把整行铺成和下拉 block 相同的白色卡片，
   按钮（灰底）落在卡片内自然形成对比；flex-end + margin 让按钮和下拉输入框
   本体底部对齐（下拉 block 自带 --block-padding 内边距）。 */
.voice-refresh-row {
  align-items: flex-end !important;
  background: var(--block-background-fill, #fff);
  border-radius: var(--block-radius, 8px);
}
.voice-refresh-row button {
  height: var(--size-10, 40px);
  /* 字面量，不能用 var(--block-padding)：该主题变量是双值 "10px 12px"，
     代入 margin 简写会让整条声明非法被丢弃。11px = block 下内边距 10px + 边框，
     让按钮和下拉输入框本体上下沿精确对齐。 */
  margin: 0 12px 11px 0;
  border-radius: var(--radius-lg) !important;
}
"""

# Gradio 的 Audio 播放器（WaveSurfer）换音频时复用同一实例，旧的播放进度会
# 原样带到新音频上（生成完成后进度条停在上一次的位置）。value=None 清空也挡
# 不住。修法：每个播放器宿主挂一次 loadedmetadata（每次换源触发、用户 seek
# 不触发），换源后开一个 ~3s 守护窗，暂停状态下把 shadow DOM 里 wavesurfer 的
# <audio>.currentTime 清回 0；窗口内用户 pointerdown（捕获阶段，躲开内部
# stopPropagation）立即取消守护，不干扰手动 seek/播放。
_RESET_AUDIO_SEEK_JS = """
() => {
  if (window.__audiocppSeekReset) return;
  window.__audiocppSeekReset = true;
  const seen = new WeakSet();
  const arm = (host) => {
    if (seen.has(host)) return;
    seen.add(host);
    const attach = () => {
      const a = host.shadowRoot && host.shadowRoot.querySelector('audio');
      if (!a) { setTimeout(attach, 250); return; }
      const guard = () => {
        let tries = 0, cancelled = false;
        const cancel = () => { cancelled = true; };
        host.addEventListener('pointerdown', cancel, { once: true, capture: true });
        const tick = () => {
          if (cancelled) return;
          if (a.paused && a.currentTime > 0.05) { try { a.currentTime = 0; } catch (e) {} }
          if (++tries < 12) setTimeout(tick, 250);
          else host.removeEventListener('pointerdown', cancel, { capture: true });
        };
        setTimeout(tick, 100);
      };
      a.addEventListener('loadedmetadata', guard);
      if (a.readyState >= 1) guard();
    };
    attach();
  };
  const scan = (root) => root.querySelectorAll('#waveform > div').forEach(arm);
  new MutationObserver((muts) => {
    for (const m of muts) {
      for (const n of m.addedNodes) {
        if (n.nodeType !== 1) continue;
        if (n.matches && n.matches('#waveform > div')) arm(n);
        else if (n.querySelectorAll) scan(n);
      }
    }
  }).observe(document.body, { childList: true, subtree: true });
  scan(document);
}
"""


with gr.Blocks(title="audio.cpp WebUI") as demo:
    with gr.Row():
        gr.Markdown("# 🎙️ audio.cpp WebUI")
        # Language names stay in their own language (English/中文), the one label a
        # user who cannot read the current UI still has to be able to find.
        lang_picker = gr.Dropdown(
            label=t("label.ui_language"), choices=language_choices(),
            value=UI_LANG, filterable=False, scale=0, min_width=170)
    status = gr.Markdown(server_status())
    with gr.Accordion(t("acc.api_usage"), open=False):
        gr.Markdown(_api_usage_md())
    with gr.Accordion(t("acc.download_settings"), open=False):
        with gr.Row():
            hf_token = gr.Textbox(
                label=t("label.hf_token"), type="password",
                placeholder=t("ph.hf_token"))
            proxy = gr.Textbox(
                label=t("label.proxy"),
                placeholder="http://127.0.0.1:7890")

    # ---- 每个标签页共用的“模型管理”卡片、接线与高级参数渲染 ----
    def _model_manager_block(task_label, tasks):
        """标准“模型管理”卡片：模型下拉 + 加载/刷新/下载/进度按钮 + 状态区。
        返回组件 dict；接线见 _wire_model_manager（刷新按钮统一接在文件末尾）。"""
        choices = choices_for_tasks(tasks)
        with gr.Group():
            gr.Markdown(t("sec.model_mgmt"))
            model = i18n_register(
                gr.Dropdown(label=t("label.model_list", task=task_label),
                            choices=choices,
                            value=(choices[0][1] if choices else None)),
                "label", "label.model_list", task=task_label)
            with gr.Row(elem_classes="mm-btn-row"):
                load_btn = gr.Button(t("btn.load"), variant="primary",
                                     size="lg", min_width=100)
                refresh_btn = gr.Button(t("btn.refresh"), variant="primary",
                                        size="lg", min_width=100)
                dl_btn = gr.Button(t("btn.download"), variant="primary",
                                   size="lg", min_width=100)
                dl_stat_btn = gr.Button(t("btn.dl_progress"), variant="primary",
                                        size="lg", min_width=100)
                unload_btn = gr.Button(t("btn.unload"), variant="secondary",
                                       size="lg", min_width=100)
            load_status = gr.Markdown("")
            dl_status = gr.Markdown("")
            timer = gr.Timer(3, active=False)
        return {"model": model, "load_btn": load_btn, "refresh_btn": refresh_btn,
                "dl_btn": dl_btn, "dl_stat_btn": dl_stat_btn, "unload_btn": unload_btn,
                "load_status": load_status, "dl_status": dl_status, "timer": timer}

    def _wire_model_manager(mm, tasks, hint):
        mm["load_btn"].click(_make_load_handler(tasks), mm["model"],
                             [mm["load_status"], status])
        mm["dl_btn"].click(download_start, [mm["model"], hf_token, proxy],
                           [mm["dl_status"], mm["timer"]])
        mm["timer"].tick(download_status_tick, mm["model"],
                         [mm["dl_status"], mm["timer"]])
        mm["dl_stat_btn"].click(download_status, mm["model"], mm["dl_status"])
        mm["unload_btn"].click(unload_model, None, [mm["load_status"], status])
        mm["model"].change(model_hint_for, mm["model"], hint)

    def _render_param_controls(model_comp, state_comp, skip=(), prefill_comp=None):
        """“高级参数”折叠区内容：按所选模型的 family 动态生成控件（gr.render），
        控件值写进共享 state（只有用户改过的项会随请求发送）。
        传入 prefill_comp（gr.State dict）时它也是渲染输入：程序化回填
        （如『🔍 分析源音频』）写 prefill 触发重渲染，把值显示在控件上。"""
        inputs = [model_comp] if prefill_comp is None else [model_comp, prefill_comp]

        @gr.render(inputs=inputs)
        def _render(model_id, prefill=None):
            specs = [p for p in params_for(model_id) if p.get("name") not in skip]
            if not specs:
                gr.Markdown(t("msg.no_advanced_params"))
                return
            prefill = prefill or {}
            for p in specs:
                if p["name"] in prefill:
                    p = {**p, "default": prefill[p["name"]]}
                comp = _make_param_component(p)
                comp.change(_adv_updater(p["name"]), [state_comp, comp], state_comp)

    # ---------------- TTS / 声音克隆 ----------------
    with gr.Tab(t("tab.tts")):
        with gr.Row(equal_height=False):
            # 左列：模型管理 + 参考音频（声音克隆）
            with gr.Column(scale=1):
                tts_mm = _model_manager_block("tts", TTS_TASKS)
                tts_model = tts_mm["model"]

                with gr.Group():
                    gr.Markdown(t("sec.reference_audio"))
                    with gr.Row(elem_classes="voice-refresh-row"):
                        tts_builtin = gr.Dropdown(
                            label=t("label.builtin_voice"),
                            choices=["(none)"] + builtin_voices(),
                            value="(none)", scale=8)
                        tts_voice_refresh = gr.Button(
                            "🔄", scale=1, min_width=48)
                    tts_upload = gr.Audio(
                        label=t("label.upload_reference_voice"), type="filepath",
                        elem_classes="audio-default")
                    tts_ref_text = gr.Textbox(
                        label=t("label.reference_text"), lines=2,
                        placeholder=t("ph.reference_text"))

                # 切换模型后的提示，显示在参考音频整块下面
                tts_hint = gr.Markdown(model_hint_for(tts_model.value))

            # 右列：合成设置 + 合成内容 + 生成 + 输出
            with gr.Column(scale=1):
                with gr.Group():
                    gr.Markdown(t("sec.tts_settings"))
                    with gr.Row():
                        tts_seed = gr.Number(label=t("label.seed"), value=1234, precision=0)
                        tts_maxtok = gr.Number(label=t("label.max_tokens"), value=1200,
                                               precision=0)
                    tts_adv_state = gr.State({})
                    with gr.Accordion(t("acc.advanced"), open=False):
                        _render_param_controls(tts_model, tts_adv_state)

                    with gr.Accordion(t("acc.other_json"), open=False):
                        tts_adv = gr.Textbox(
                            label="",
                            placeholder='{"num_inference_steps": 10, "voice_samples": "D:/a.wav,D:/b.wav"}',
                            lines=3)

                with gr.Group():
                    gr.Markdown(t("sec.tts_content"))
                    tts_text = gr.Textbox(
                        label=t("label.text_to_speak"), lines=5,
                        value=t("value.tts_sample_text"))
                    tts_lang = gr.Dropdown(
                        label=t("label.language_blank_default"),
                        choices=tts_language_choices(), value="english")

                tts_btn = gr.Button(t("btn.tts"), variant="primary", size="lg")
                with gr.Group():
                    gr.Markdown(t("sec.output_audio"))
                    tts_out = gr.Audio(label=t("label.output_audio"), type="filepath",
                                       elem_classes="audio-default")
                    tts_msg = gr.Markdown("")

        _wire_model_manager(tts_mm, TTS_TASKS, tts_hint)
        tts_model.change(lambda: {}, None, tts_adv_state)  # reset knobs on model switch
        tts_builtin.change(on_builtin_voice_change, tts_builtin,
                           [tts_upload, tts_ref_text])
        tts_voice_refresh.click(refresh_builtin_voices, tts_builtin,
                                [tts_builtin, tts_upload, tts_ref_text])
        # Clear the previous run's audio + status message the moment 生成 is
        # clicked, so a prior message doesn't linger next to the new run's
        # progress indicator (outputs otherwise update only when do_tts returns).
        tts_btn.click(lambda: (None, ""), None, [tts_out, tts_msg]).then(
            do_tts,
            [tts_model, tts_text, tts_lang, tts_upload, tts_builtin,
             tts_ref_text, tts_seed, tts_maxtok, tts_adv_state, tts_adv],
            [tts_out, tts_msg])

    # ---------------- ASR / 音频转写 ----------------
    with gr.Tab(t("tab.asr")):
        with gr.Row(equal_height=False):
            # 左列：模型管理 + 音频输入
            with gr.Column(scale=1):
                asr_mm = _model_manager_block("asr", ASR_TASKS)
                asr_model = asr_mm["model"]

                with gr.Group():
                    gr.Markdown(t("sec.audio_input"))
                    asr_audio = gr.Audio(label=t("label.upload_or_record"), type="filepath",
                                         elem_classes="audio-default")
                with gr.Accordion(t("acc.asr_options"), open=False):
                    asr_language = gr.Dropdown(
                        label=t("label.force_language"),
                        choices=asr_language_choices(), value="")
                    asr_context = gr.Textbox(
                        label=t("label.context_hint"), lines=2,
                        placeholder=t("ph.context_hint"))
                    asr_dialogue = gr.Checkbox(
                        label=t("label.dialogue_mode"))
                asr_hint = gr.Markdown(model_hint_for(asr_model.value))
                asr_btn = gr.Button(t("btn.asr"), variant="primary", size="lg")

            with gr.Column(scale=1):
                with gr.Group():
                    gr.Markdown(t("sec.asr_result"))
                    asr_out = gr.Textbox(
                        label=t("label.transcript"), lines=10,
                        placeholder=t("ph.transcript"))
                    asr_msg = gr.Markdown("")

        _wire_model_manager(asr_mm, ASR_TASKS, asr_hint)
        asr_btn.click(lambda: ("", ""), None, [asr_out, asr_msg]).then(
            do_asr, [asr_model, asr_audio, asr_language, asr_context, asr_dialogue],
            [asr_out, asr_msg])

    # ---------------- 音乐 / 音效生成 ----------------
    with gr.Tab(t("tab.gen")):
        with gr.Row(equal_height=False):
            # 左列：模型管理 + 源音频（编辑类用法）
            with gr.Column(scale=1):
                gen_mm = _model_manager_block("gen", GEN_TASKS)
                gen_model = gen_mm["model"]

                with gr.Group():
                    gr.Markdown(t("sec.gen_source"))
                    gen_audio = gr.Audio(
                        label=t("label.upload_source_for_edit"),
                        type="filepath", elem_classes="audio-default")
                    gen_ana_btn = gr.Button(
                        t("btn.analyze_source"))
                    gen_ana_msg = gr.Markdown("")

                gen_hint = gr.Markdown(model_hint_for(gen_model.value))

            # 右列：生成设置 + 提示词/歌词 + 生成 + 输出
            with gr.Column(scale=1):
                with gr.Group():
                    gr.Markdown(t("sec.gen_settings"))
                    with gr.Row():
                        gen_duration = gr.Number(
                            label=t("label.duration_seconds"), value=30, precision=1)
                        gen_seed = gr.Number(label=t("label.seed"), value=1234, precision=0)
                    gen_adv_state = gr.State({})
                    gen_prefill = gr.State({})   # 『🔍 分析源音频』回填 -> 触发控件重渲染
                    with gr.Accordion(t("acc.advanced"), open=False):
                        _render_param_controls(gen_model, gen_adv_state,
                                               prefill_comp=gen_prefill)

                    with gr.Accordion(t("acc.other_json"), open=False):
                        gen_adv = gr.Textbox(
                            label="",
                            placeholder='{"tags": "pop,bright,drums", "num_inference_steps": 8}',
                            lines=3)

                with gr.Group():
                    gr.Markdown(t("sec.prompt_lyrics"))
                    gen_text = gr.Textbox(
                        label=t("label.prompt"), lines=3,
                        value="uplifting pop with bright synths and driving drums")
                    gen_lyrics = gr.Textbox(
                        label=t("label.lyrics_optional"), lines=4)

                gen_btn = gr.Button(t("btn.gen"), variant="primary", size="lg")
                with gr.Group():
                    gr.Markdown(t("sec.output_audio"))
                    gen_out = gr.Audio(label=t("label.output_audio"), type="filepath",
                                       elem_classes="audio-default")
                    gen_msg = gr.Markdown("")

        _wire_model_manager(gen_mm, GEN_TASKS, gen_hint)
        gen_model.change(lambda: ({}, {}, ""), None,
                         [gen_adv_state, gen_prefill, gen_ana_msg])  # 切换模型清空旋钮+分析信息
        gen_ana_btn.click(lambda: t("msg.analyzing", load=t("btn.load")),
                          None, gen_ana_msg).then(
            do_music_analyze, [gen_model, gen_audio, gen_seed, gen_adv_state],
            [gen_adv_state, gen_prefill, gen_ana_msg])
        gen_btn.click(lambda: (None, ""), None, [gen_out, gen_msg]).then(
            do_music_gen,
            [gen_model, gen_text, gen_lyrics, gen_audio, gen_duration, gen_seed,
             gen_adv_state, gen_adv],
            [gen_out, gen_msg])

    # ---------------- 声音转换 (vc / svc / s2s) ----------------
    with gr.Tab(t("tab.vc")):
        with gr.Row(equal_height=False):
            # 左列：模型管理 + 源音频 + 目标音色
            with gr.Column(scale=1):
                vc_mm = _model_manager_block("vc/svc/s2s", VC_TASKS)
                vc_model = vc_mm["model"]

                with gr.Group():
                    gr.Markdown(t("sec.vc_source"))
                    vc_source = gr.Audio(label=t("label.upload_source"), type="filepath",
                                         elem_classes="audio-default")
                with gr.Group():
                    gr.Markdown(t("sec.vc_target"))
                    with gr.Row(elem_classes="voice-refresh-row"):
                        vc_builtin = gr.Dropdown(
                            label=t("label.builtin_voice"),
                            choices=["(none)"] + builtin_voices(),
                            value="(none)", scale=8)
                        vc_voice_refresh = gr.Button(
                            "🔄", scale=1, min_width=48)
                    vc_target = gr.Audio(label=t("label.upload_target_voice"), type="filepath",
                                         elem_classes="audio-default")

                vc_hint = gr.Markdown(model_hint_for(vc_model.value))

            # 右列：转换设置 + 转换 + 输出
            with gr.Column(scale=1):
                with gr.Group():
                    gr.Markdown(t("sec.vc_settings"))
                    vc_seed = gr.Number(label=t("label.seed"), value=1234, precision=0)
                    vc_adv_state = gr.State({})
                    with gr.Accordion(t("acc.advanced"), open=False):
                        _render_param_controls(vc_model, vc_adv_state)
                    with gr.Accordion(t("acc.other_json"), open=False):
                        vc_adv = gr.Textbox(
                            label="",
                            placeholder='{"route": "style_converted_vc", "style_ref": "D:/style.wav", "target_text": "……"}',
                            lines=3)

                vc_btn = gr.Button(t("btn.vc"), variant="primary", size="lg")
                with gr.Group():
                    gr.Markdown(t("sec.output_audio"))
                    vc_out = gr.Audio(label=t("label.output_audio"), type="filepath",
                                      elem_classes="audio-default")
                    vc_msg = gr.Markdown("")

        _wire_model_manager(vc_mm, VC_TASKS, vc_hint)
        vc_model.change(lambda: {}, None, vc_adv_state)  # reset knobs on model switch
        vc_builtin.change(lambda n: on_builtin_voice_change(n)[0], vc_builtin, vc_target)
        vc_voice_refresh.click(lambda n: refresh_builtin_voices(n)[:2],
                               vc_builtin, [vc_builtin, vc_target])
        vc_btn.click(lambda: (None, ""), None, [vc_out, vc_msg]).then(
            do_vc,
            [vc_model, vc_source, vc_target, vc_builtin, vc_seed,
             vc_adv_state, vc_adv],
            [vc_out, vc_msg])

    # ---------------- 音源分离 (sep) ----------------
    with gr.Tab(t("tab.sep")):
        with gr.Row(equal_height=False):
            # 左列：模型管理 + 输入音频
            with gr.Column(scale=1):
                sep_mm = _model_manager_block("sep", SEP_TASKS)
                sep_model = sep_mm["model"]

                with gr.Group():
                    gr.Markdown(t("sec.sep_input"))
                    sep_audio = gr.Audio(label=t("label.upload_to_separate"), type="filepath",
                                         elem_classes="audio-default")
                sep_hint = gr.Markdown(model_hint_for(sep_model.value))
                sep_btn = gr.Button(t("btn.sep"), variant="primary", size="lg")

            # 右列：分离结果（各分轨播放器 + 文件下载）
            with gr.Column(scale=1):
                with gr.Group():
                    gr.Markdown(t("sec.sep_result"))
                    sep_stems = [gr.Audio(label=t("label.stem_n", n=i + 1), type="filepath",
                                          visible=False, elem_classes="audio-default")
                                 for i in range(MAX_SEP_STEMS)]
                    sep_files = gr.File(label=t("label.all_stem_files"), file_count="multiple",
                                        interactive=False)
                    sep_msg = gr.Markdown("")

        _wire_model_manager(sep_mm, SEP_TASKS, sep_hint)
        sep_btn.click(
            lambda: (*(gr.update(value=None, visible=False)
                       for _ in range(MAX_SEP_STEMS)), None, ""),
            None, [*sep_stems, sep_files, sep_msg]).then(
            do_sep, [sep_model, sep_audio], [*sep_stems, sep_files, sep_msg])

    # ---------------- 音频分析 (vad / diar / align) ----------------
    with gr.Tab(t("tab.analyze")):
        with gr.Row(equal_height=False):
            # 左列：模型管理 + 音频输入（align 模型多出对齐文本/语言）
            with gr.Column(scale=1):
                ana_mm = _model_manager_block("vad/diar/align", ANALYZE_TASKS)
                ana_model = ana_mm["model"]

                with gr.Group():
                    gr.Markdown(t("sec.audio_input"))
                    ana_audio = gr.Audio(label=t("label.upload_or_record"), type="filepath",
                                         elem_classes="audio-default")
                    gr.Markdown(
                        t("msg.auto_16k_mono"),
                        elem_classes="hint-small")
                    _ana_is_align = bool(
                        ana_model.value and
                        (catalog_by_id(ana_model.value) or {}).get("task") == "align")
                    ana_text = gr.Textbox(
                        label=t("label.align_text"), lines=3,
                        visible=_ana_is_align)
                    ana_lang = gr.Textbox(
                        label=t("label.language_optional"),
                        visible=_ana_is_align)
                ana_hint = gr.Markdown(model_hint_for(ana_model.value))
                ana_btn = gr.Button(t("btn.analyze"), variant="primary", size="lg")

            # 右列：分析结果
            with gr.Column(scale=1):
                with gr.Group():
                    gr.Markdown(t("sec.analyze_result"))
                    ana_out = gr.Textbox(
                        label=t("label.analyze_output"), lines=14,
                        placeholder=t("ph.analyze_output"))
                    ana_json = gr.File(label=t("label.raw_json"), interactive=False)
                    ana_msg = gr.Markdown("")

        _wire_model_manager(ana_mm, ANALYZE_TASKS, ana_hint)
        ana_model.change(_align_fields_visibility, ana_model, [ana_text, ana_lang])
        ana_btn.click(lambda: ("", None, ""), None, [ana_out, ana_json, ana_msg]).then(
            do_analyze, [ana_model, ana_audio, ana_text, ana_lang],
            [ana_out, ana_json, ana_msg])

    # ---------------- 声音设计 (vdes) ----------------
    with gr.Tab(t("tab.vdes")):
        with gr.Row(equal_height=False):
            # 左列：模型管理 + 生成设置
            with gr.Column(scale=1):
                vdes_mm = _model_manager_block("vdes", VDES_TASKS)
                vdes_model = vdes_mm["model"]

                with gr.Group():
                    gr.Markdown(t("sec.gen_settings"))
                    with gr.Row():
                        vdes_seed = gr.Number(label=t("label.seed"), value=1234, precision=0)
                        vdes_maxtok = gr.Number(label="max_tokens", value=1200, precision=0)
                    vdes_adv_state = gr.State({})
                    with gr.Accordion(t("acc.advanced"), open=False):
                        # instruct 有专用的『音色描述』输入框，这里不重复生成
                        _render_param_controls(vdes_model, vdes_adv_state, skip=("instruct",))
                    with gr.Accordion(t("acc.other_json"), open=False):
                        vdes_adv = gr.Textbox(label="", placeholder='{"temperature": 0.9}',
                                              lines=3)
                vdes_hint = gr.Markdown(model_hint_for(vdes_model.value))

            # 右列：设计内容 + 生成 + 输出
            with gr.Column(scale=1):
                with gr.Group():
                    gr.Markdown(t("sec.vdes_content"))
                    vdes_instruct = gr.Textbox(
                        label=t("label.voice_description"), lines=3,
                        placeholder=t("ph.voice_description"))
                    vdes_text = gr.Textbox(
                        label=t("label.text_to_speak"), lines=5,
                        value=t("value.vdes_sample_text"))

                vdes_btn = gr.Button(t("btn.vdes"), variant="primary", size="lg")
                with gr.Group():
                    gr.Markdown(t("sec.output_audio"))
                    vdes_out = gr.Audio(label=t("label.output_audio"), type="filepath",
                                        elem_classes="audio-default")
                    vdes_msg = gr.Markdown("")

        _wire_model_manager(vdes_mm, VDES_TASKS, vdes_hint)
        vdes_model.change(lambda: {}, None, vdes_adv_state)  # reset knobs on model switch
        vdes_btn.click(lambda: (None, ""), None, [vdes_out, vdes_msg]).then(
            do_vdes,
            [vdes_model, vdes_text, vdes_instruct, vdes_seed, vdes_maxtok,
             vdes_adv_state, vdes_adv],
            [vdes_out, vdes_msg])

    # 上传/录制后把文件换成短 ASCII 临时名，绕过 Gradio 在 Windows 上无法渲染
    # 含中文/超长文件名的声波图（见 _stage_upload）。只接输入类音频控件。
    for _in_audio in (tts_upload, asr_audio, gen_audio, vc_source, vc_target,
                      sep_audio, ana_audio):
        # upload: 长文件强制换成新的稳定临时文件，避免 X 清空后再次上传长音频时
        # 前端复用旧 preview 状态导致波形不渲染。
        _in_audio.upload(lambda p: _stage_upload(p, force_copy=True), _in_audio, _in_audio)
        # microphone: 录音完成不是 upload 事件，单独接 stop_recording。
        _in_audio.stop_recording(_stage_recording, _in_audio, _in_audio)
        # clear: 明确把后端值置空，避免长音频 preview fetch 被中断后控件状态残留。
        _in_audio.clear(lambda: None, None, _in_audio)

    # 顺序与 refresh()/TAB_SPECS 一致：各页下拉、状态行、各页提示。
    _refresh_outputs = [
        tts_model, asr_model, gen_model, vc_model, sep_model, ana_model, vdes_model,
        status,
        tts_hint, asr_hint, gen_hint, vc_hint, sep_hint, ana_hint, vdes_hint,
    ]
    for _mm in (tts_mm, asr_mm, gen_mm, vc_mm, sep_mm, ana_mm, vdes_mm):
        _mm["refresh_btn"].click(refresh, None, _refresh_outputs)

    # --- language picker -----------------------------------------------------
    # Explicit entries were appended while the tree was built; keep them and add
    # everything reverse lookup can find on top.
    _i18n_registry.extend(build_i18n_registry(demo.blocks.values()))
    _relabel_comps, _ = _relabel_updates()
    _model_dropdowns = [tts_model, asr_model, gen_model, vc_model,
                        sep_model, ana_model, vdes_model]
    _hints = [tts_hint, asr_hint, gen_hint, vc_hint, sep_hint, ana_hint, vdes_hint]

    # Sample texts are defaults the user is free to overwrite, so they are localized
    # only while still untouched — see _sample_text_update.
    _samples = [(tts_text, "value.tts_sample_text"),
                (vdes_text, "value.vdes_sample_text")]

    def _is_untouched_sample(current, key):
        """True if the box still holds the sample text we put there.

        Distinguishing that from something the user typed is what keeps a language
        switch from throwing their input away. Checked before switching, so `key`
        still resolves in the outgoing language.
        """
        return not (current or "").strip() or (current or "").strip() == t(key).strip()

    def switch_language(lang, asr_lang, tts_lang, *rest):
        """Re-localize the whole UI in place, preserving what the user picked.

        Static labels come from the registry. Anything whose text is computed at
        request time must be rebuilt by hand: the model dropdowns (their labels carry
        the catalog's localized display_name plus "not installed"/VRAM notes), the
        per-tab hints, the status line, and both language pickers, whose labels are
        localized while their values stay the English names the server parses.
        """
        sample_values, selected = rest[:len(_samples)], rest[len(_samples):]
        # Decide before switching, while the keys still resolve in the old language.
        untouched = [_is_untouched_sample(v, k)
                     for v, (_, k) in zip(sample_values, _samples)]
        effective = set_language(lang)
        sample_updates = [gr.update(value=t(k)) if fresh else gr.update()
                          for fresh, (_, k) in zip(untouched, _samples)]
        _, updates = _relabel_updates()

        dropdowns, hints = [], []
        for tasks, current in zip(TAB_SPECS, selected):
            choices = choices_for_tasks(tasks)
            # Keep the selection across the switch; only its label changed.
            keep = current if any(c[1] == current for c in choices) else (
                choices[0][1] if choices else None)
            dropdowns.append(gr.update(choices=choices, value=keep))
            hints.append(model_hint_for(keep))

        return [gr.update(label=t("label.ui_language"), value=effective),
                gr.update(choices=asr_language_choices(), value=asr_lang),
                gr.update(choices=tts_language_choices(), value=tts_lang),
                *sample_updates, *updates, *dropdowns, server_status(), *hints]

    lang_picker.change(
        switch_language,
        [lang_picker, asr_language, tts_lang, *[c for c, _ in _samples],
         *_model_dropdowns],
        [lang_picker, asr_language, tts_lang, *[c for c, _ in _samples],
         *_relabel_comps, *_model_dropdowns, status, *_hints])
    # 顶部状态行在建界面时只求值一次，浏览器刷新会看到那份静态初值（即使模型
    # 已加载也显示“server 未运行”）。server 本身就是状态的唯一真相源
    # （/health + /v1/models），每次页面加载重新探测即可，无需额外状态文件。
    demo.load(server_status, None, status)
    # 音频播放器换源后把旧播放进度清回 0:00（见 _RESET_AUDIO_SEEK_JS 注释）。
    demo.load(None, None, None, js=_RESET_AUDIO_SEEK_JS)
    gr.Markdown(
        "---\n<center><small>" + t("footer") + "</small></center>")


if __name__ == "__main__":
    open_browser = os.environ.get("AUDIOCPP_NO_BROWSER") != "1"
    # Gradio 6 moved css/theme from the Blocks constructor to launch().
    demo.launch(server_name="127.0.0.1", server_port=7860, inbrowser=open_browser,
                css=CUSTOM_CSS)
