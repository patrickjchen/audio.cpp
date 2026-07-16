# audio.cpp WebUI

A Gradio frontend (`webui.py`) that proxies to the local `audiocpp_server` HTTP API.
It loads models on demand: pick a model in the UI and click load, and the WebUI
(re)starts `audiocpp_server` with a single-model config. One model lives in VRAM at a
time — picking a different model swaps it. No need to start a server yourself.

## Credit

The WebUI originates from **[kigner/audio.cpp-webui](https://github.com/kigner/audio.cpp-webui)**
and is the bulk of the logic here — the Gradio UI, on-demand single-model loading,
model management and background downloads, the per-family request profiles, the VRAM
budgeting and adaptive chunking for TTS/ASR/voice conversion, and the dialogue mode
that chains diarization into ASR.

That fork runs on Windows only and its interface is Chinese only. What was added here
is cross-platform support (Linux/macOS binary and build discovery, `run_webui.sh`) and
the localization layer, with English as the default and the original Chinese kept as a
locale — see [UI language](#ui-language).

## Launching

```bash
./run_webui.sh          # from the repo root
```

Then open **http://127.0.0.1:7860**.

`run_webui.sh` only locates a Python that has the dependencies and executes
`webui/webui.py`; all detection (backend, server binary, bundle root) happens in
`webui.py`. It tries, in order: `$AUDIOCPP_PYTHON`, `venv/bin/python`,
`.venv/bin/python`, `python3` on `PATH`, `python` on `PATH`. If none is found it
prints the setup command and exits:

```bash
python3 -m venv venv
./venv/bin/pip install gradio requests torch safetensors pyyaml huggingface_hub
```

The web UI on 7860 is for humans. To serve **other programs**, run `audiocpp_server`
directly — see the server section of the root [`README.md`](../README.md) for its
config and endpoints.

## Conventions

- **Models are named by catalog id.** `configs/models_catalog.json` maps each `id` to
  its `family` / `task` / `path` / `download_id`, so you never write those by hand.
  Entries whose model directory is missing show as *not installed*; click Download
  Model in the UI to fetch them in the background, or run
  `python tools/model_manager.py install <download_id> --models-root <bundle>/models`.
- **Backend auto-detection.** `AUDIOCPP_BACKEND=gpu` (=cuda) or `cpu` wins if set.
  Otherwise the WebUI picks `gpu` when an NVIDIA driver (`nvidia-smi`, or `nvcuda.dll`
  on Windows) **and** a gpu server build are both present, else `cpu`. The CPU backend
  works, just slower and with lower model coverage; on it, ggml threads default to all
  cores minus one (override with `AUDIOCPP_THREADS=N`).
- **Server binary discovery.** From-source builds are found at
  `build/<os>-<backend>-<type>/bin` (e.g. `build/linux-cuda-release/bin`), where the
  backend is in the directory name, newest wins per backend. A plain `cmake -B build`
  lands in `build/bin` and says nothing about the backend, so that layout is classified
  from its `CMakeCache.txt` (`GGML_CUDA:BOOL=ON` → gpu). A packaged bundle (a root
  holding `cpu/ gpu/ models/`) is used when present; override the bundle root with
  `AUDIOCPP_BUNDLE`.
- **Long text needs no separate script.** The TTS tab splits long input into chunks,
  synthesizes them one by one, and concatenates the result into a single wav. The budget
  is per-family: VibeVoice 600 characters/chunk, VoxCPM2 60 (its audiovae decode graph
  grows with chunk length on 8G cards), 1000 for everything else. The CLI equivalent is
  `audiocpp_cli --batch-text-file <txt> --batch-merge-audio concat`.
- **Relative paths** inside the UI (e.g. `voice/demo_01_man.wav`, `output/xxx.wav`) are
  relative to the `webui/` directory.

## UI language

Pick the language from the dropdown at the top of the page; it re-labels the UI
immediately, no restart. To choose the language it starts in:

```bash
AUDIOCPP_LANG=zh ./run_webui.sh     # default is en
```

Strings live in `locales/<lang>.json`, one file per language. Shipped: `en.json`
(the source of truth) and `zh.json`.

- `en.json` is the fallback for any key a translation is missing, so a partial
  translation degrades to English rather than to a raw key.
- **Adding a language** means dropping in a new `locales/<code>.json` — no code
  change. It appears in the dropdown on the next start. Give it a self-name in
  `_LANG_SELF_NAMES` in `webui.py` to be listed as something other than its code.
  An unknown or malformed locale logs a warning and falls back to English.
- Catalog entries in `configs/models_catalog.json` carry English `display_name` /
  `input_hint`. A locale may override any entry with `catalog.<id>.display_name` /
  `catalog.<id>.input_hint` keys — this is how `zh.json` restores the Chinese model
  names.
- Switching is **process-wide, not per browser session**: Gradio serves one Blocks
  object to every connection, so there is nowhere to hang per-session labels. Fine
  for a local single-user tool; a second open tab will follow along.
- Sample texts (the TTS and Voice Design boxes) are only re-localized while still
  untouched, so a switch never discards something you typed.
- Values the server parses stay English no matter the language: the language pickers
  and `max_tokens` show a localized label over an unchanged wire value.

Gradio's own widget text — "Click to Upload", "Submit", error toasts — is **not**
covered by these files. It is compiled into Gradio's frontend bundle and selected
from the browser's locale, so it follows the browser rather than this dropdown: a
zh-CN browser already renders it in Chinese (`点击上传`), an English one does not.
Forcing it would mean lying about `navigator.language` before the bundle loads, and
it could then only change on a page reload, so it is left alone.

---

## WebUI advanced parameters (widgets generated per model)

The controls under *Synthesis settings → Advanced parameters* on the TTS tab are driven
by **`configs/model_params.json`**: when you select a model, the WebUI dynamically
generates the matching sliders / number boxes / toggles / text boxes (`gr.render`) for
its `family` — no hand-written JSON needed. Below the widgets there is a collapsible
*Other parameters (JSON)* box as an escape hatch for keys the config doesn't list.
General rules:

- **Only widget values you actually changed are sent with the request** (untouched ones
  use the model's own defaults). `options` merge order: family defaults → generated
  widgets → JSON box (JSON wins over widgets).
- `seed` and `max_tokens` have dedicated inputs (synthesis settings) and are not
  repeated here.
- Use Upload/Record or the built-in reference voices for the reference timbre; put the
  transcript of that audio in the *Reference text* box (equivalent to `reference_text`).
- Bad values are usually ignored, or the server errors out — errors are shown **below
  the output audio** (no popup card).
- **Chatterbox** clone parameters are fixed at model load time: after changing them you
  must click *📥 Load model* again for them to take effect (otherwise you get
  "session config is fixed").

### Custom widgets (edit `configs/model_params.json`)

Grouped by `family`, one widget spec per entry:

```json
{"name": "guidance_scale", "type": "slider", "label": "guidance_scale",
 "default": 1.3, "minimum": 0.0, "maximum": 5.0, "step": 0.1, "info": "CFG guidance strength"}
```

- `name`: the key passed through to the request `options`. `type`: `slider` / `number`
  (`precision:0` for integers) / `bool` / `text` / `choice` (with `choices:[...]`).
- `default` should equal the model's own default (checked against each
  `src/models/<family>/*.cpp`).
- Click *🔄 Refresh list* in the UI to reload this file — no restart needed.
- File-path and parity-style rare parameters (e.g. `*_noise_file`) have no widget; pass
  them via the *Other parameters (JSON)* box. Quantization keys (e.g.
  `vibevoice.weight_type`) are covered in the root `README.md`.

The table below lists the complete set of keys each model's `session.cpp` **actually
reads** (widgets are a curated common subset; the rest still work through the JSON box):

| Model (family) | Available keys (also usable in the JSON box) | Example |
|---|---|---|
| **Qwen3-TTS** (qwen3_tts) 0.6B / 1.7B / CustomVoice | `do_sample` `temperature` `top_k` `top_p`; CustomVoice builds also take `speaker` | `{"do_sample": true, "temperature": 0.8, "top_k": 40, "top_p": 0.9}`<br>CustomVoice built-in voice: `{"speaker": "<CustomVoice voice name>"}` |
| **VibeVoice** (vibevoice) 1.5B long-form/multi-speaker | `num_inference_steps` `guidance_scale` `max_length_times` `do_sample` `temperature` `top_k` `top_p`; multi-speaker `voice_samples` (comma-separated wavs, max 4, **cannot** be combined with a reference voice) | `{"num_inference_steps": 10, "guidance_scale": 1.3, "max_length_times": 2.0}`<br>Multi-speaker: `{"voice_samples": "/path/to/a.wav,/path/to/b.wav"}` |
| **VoxCPM2** (voxcpm2) | `num_inference_steps` `guidance_scale` `min_tokens` `retry_badcase` `retry_badcase_max_times` `retry_badcase_ratio_threshold`; reference transcript `prompt_text` | `{"num_inference_steps": 10, "guidance_scale": 2.0, "retry_badcase": true}` |
| **MioTTS** (miotts, needs MioCodec) | `temperature` `top_k` `top_p` `repetition_penalty` `presence_penalty` `frequency_penalty` `do_sample` `best_of_n` `best_of_n_enabled` `best_of_n_language` | `{"temperature": 0.9, "top_p": 0.9, "repetition_penalty": 1.1, "best_of_n": 3}` |
| **Chatterbox** (chatterbox, voice cloning) | `exaggeration` `guidance_scale` `temperature` `repetition_penalty` `min_p` `top_p` `s3gen_cfg_rate` `max_new_tokens` `do_sample` `greedy` `stop_on_eos` | `{"exaggeration": 0.5, "guidance_scale": 0.5, "temperature": 0.8, "repetition_penalty": 1.2}` |
| **OmniVoice** (omnivoice) | `instruct` (style/instruction text); `reference_text` (the *Reference text* box is normally enough) | `{"instruct": "read this in a light, brisk tone"}` |
| **Pocket TTS** (pocket_tts) | No dedicated advanced parameters (reference voice + language only) | — |

> Key names come from the options each model's `src/models/<family>/session.cpp`
> actually reads; the same key may have a different range or meaning across models.
> Quantization keys (e.g. `vibevoice.weight_type`, `voxcpm2.*_weight_type`) are covered
> in the quantization section of the root `README.md` — they are not general-purpose
> defaults.

### Music generation / voice conversion parameter details

The in-page hints are trimmed; the full explanation lives here.

**ACE-Step (music generation/editing)**

- The prompt describes style/instruments/mood (English works best); lyrics are optional.
  A duration of `-1` means auto.
- `task_route` operation type: `text2music` = pure text-to-music (default, no source
  audio needed); `cover` = re-lyric cover (the main upstream Remix route, used with the
  two cover sliders below); `cover-nofsq` = cover variant that skips FSQ quantization;
  `remix` = fine-grained flow-edit re-lyric; `complete` / `lego` / `extract` / `repaint`
  are other editing routes. **Every route except text2music needs a source audio
  upload.**
- After uploading source audio, click *🔍 Analyze source audio* first: it back-infers the
  source track's caption/lyrics/BPM/key and fills them into the advanced parameters
  (especially recommended before a remix/cover re-lyric; requires *📥 Load model* first,
  and takes a few tens of seconds for one minute of audio). The analysis is
  reproducible: the same audio analyzes identically every time (VAE mean encoding; with
  seed=-1 the analysis always uses 1234 — set a concrete seed to re-roll the lyric
  transcription).
- Diffusion parameters: `num_inference_steps` is capped at 20 for turbo, and when left
  blank defaults to 16 on the remix route and 8 on the others; `shift` (timestep
  warping) defaults to 3.0 to match the upstream turbo UI — dropping it back to 1.0
  noticeably degrades remix re-lyric articulation.
- The two cover-route sliders:
  - `audio_cover_strength` (Remix strength): what fraction of the denoising steps
    reference the source structure. 1 = close to the original, 0 = free rein; upstream
    Remix suggests 0.5. Only applies to cover/cover-nofsq.
  - `cover_noise_strength` (melody retention): denoise starting from a partially noised
    source. 0 = no melody retention, 0.1–0.25 = recommended range (keeps the melody
    while allowing new lyrics/style), higher sticks closer to the original. Only applies
    to cover.
- remix (flow-edit) parameters:
  - `source_caption` / `source_lyrics`: source-side text conditioning (the source
    track's own style caption / original lyrics, with `[Verse]` `[Chorus]` tags). An
    empty caption falls back to the main prompt; *🔍 Analyze* can fill these in.
    **New lyrics go in the main *Lyrics* box.**
  - `flow_edit_n_min` (edit start): the fraction of leading high-noise steps to skip.
    0 = edit from the very beginning; raising it preserves the source more but weakens
    the re-lyric.
  - `flow_edit_n_max` (edit end): 1 = paired editing throughout; lowering it to 0.7–0.9
    makes the tail denoise toward the new lyrics only — **try this first when the lyrics
    won't come through**.
  - `flow_edit_n_avg`: samples per step, averaged (remix defaults to 2 for stability),
    1 = fastest. Note that the remix default of 16 steps × n_avg 2 is about 4× the cost
    of the old default (8 steps × 1); dial it back manually if you want speed.
- Score parameters `bpm` / `keyscale` (e.g. `F major`, `c# minor`) / `timesignature`
  (e.g. `4`): 0/blank = unspecified; *🔍 Analyze* fills them in.

**Stable Audio (music/sound effects)**: the prompt is **English only** and lyrics are
not used; the music build generates music, the sfx build generates sound effects.
Uploading source audio enables continuation/repair: set `audio_input_kind` to
`init_audio` (with `init_noise_level` for strength) or `inpaint_audio`.

**HeartMuLa (song generation from lyrics + tags)**: the advanced parameter `tags` is
required (comma-separated, e.g. `pop,bright,drums,female vocals`), and *Lyrics* holds
the sung text. A 3B model; upstream measures ~25G peak VRAM for a 120-second song
(docs/memory_saver.md), so an 8G card can't run it. mem_saver is on by default, and
`infinite_mode` can be enabled for long songs.

**Seed-VC (voice conversion)**: source speech + a target timbre reference (a few seconds
to a few tens of seconds of clean vocals). Leave `route` blank to use the task default
(vc entries → `v2_vc`, svc entries → `v1_svc`); `v1_whisper_bigvgan_vc` /
`v1_xlsr_hift_vc` are the older v1 routes; `v1_svc` only works with svc entries.
`intelligibility_cfg_rate` / `similarity_cfg_rate` only apply to v2,
`inference_cfg_rate` only to v1.

**Vevo2 (voice conversion)**: defaults to `route=style_preserved_vc` (keeps the source's
speaking style, swaps only the timbre). A blank `route` uses the entry's task default
(vc → style_preserved_vc, svc → style_preserved_svc, s2s → editing) and must match the
selected entry's task; `style_converted_*` / `editing` additionally need `style_ref`
(a server-local wav path) / `style_ref_text` / `target_text` supplied through the *Other
parameters (JSON)* box. Leave `use_pitch_shift` (global transpose by the source/target
median pitch difference) blank to use the route default: on for style_preserved_* and
the singing routes, off for style_converted_vc / editing. Long audio is adaptively
chunked by "target timbre duration + per-chunk source duration ≤ VRAM budget" and then
concatenated; reference timbre longer than about 10s is truncated automatically (an 8G
VRAM constraint).

### Per-task input requirements

- **VibeVoice**: multi-speaker scripts use one `Speaker N: content` per line (N starts
  at 0); plain text is automatically wrapped as `Speaker 0: ...`. For distinct voices per
  role use the advanced parameter `voice_samples` (comma-separated server-local wavs, ≤4)
  — in that case do **not** also upload a reference voice.
  **VibeVoice produces gibberish on very short text** (below roughly 40 CJK characters /
  35 English words). This is a model trait — no voice, parameter, or seed fixes it — so the
  WebUI rejects such input up front and asks you to lengthen the text or switch to
  `qwen3-tts` / `voxcpm2` / `pocket-tts`. For the same reason, a too-short trailing chunk
  left over by splitting is merged back into the previous chunk.
- **VoxCPM2 / Qwen3-TTS**: upload or select a clean single-speaker reference voice and
  put its transcript in the *Reference text* box, otherwise output may truncate early.
  Long text is automatically chunked and concatenated; VoxCPM2 defaults to q8_0
  quantization on 8G cards.
- **Chatterbox**: language supports only english / spanish / french / german / italian /
  portuguese / korean (no Chinese/Japanese/Russian, and no auto-detection); blank =
  English.
- **Qwen3-ASR**: long audio is automatically split at silences into ≤60-second chunks,
  transcribed, and concatenated. *Context hint* takes names/terms/background (e.g.
  "meeting about ggml quantization, attendees: …") to help recognize proper nouns.
  Conversation mode (limited to 120s) first runs Sortformer speaker diarization (≤4
  speakers), then transcribes each segment into a dialogue transcript with speakers and
  timestamps; the Sortformer model must be installed.
- **Audio analysis (VAD/separation/alignment)**: WAV input is converted to 16 kHz mono
  before being fed to the model, and result timelines are computed at 16 kHz. Qwen3
  forced alignment handles about 115 seconds of audio per call.
- **Source separation**: HTDemucs outputs four stems — drums/bass/other/vocals (slow on
  long audio); Mel-Band RoFormer outputs a vocals stem plus an accompaniment stem
  (mixture − vocals).
- **Model downloads** run in the background with auto-refreshing progress; *📊 Download
  progress* shows it on demand.

---

## Model id quick reference

The full list is in `configs/models_catalog.json` (each entry has `id` / `family` /
`path` / `task` / `download_id`). Common ones:

| id | family | task | Notes |
|---|---|---|---|
| `qwen3-tts` | qwen3_tts | tts | Qwen3-TTS 0.6B (voice cloning) |
| `qwen3-asr` | qwen3_asr | asr | Qwen3-ASR 0.6B |
| `vibevoice` | vibevoice | tts | VibeVoice 1.5B (long-form/multi-speaker, `Speaker N:` scripts) |
| `omnivoice` | omnivoice | tts | OmniVoice |
| `pocket-tts` | pocket_tts | tts | Pocket TTS (needs a reference voice) |

Not-installed ids are flagged at runtime; click Download in the WebUI, or run
`python tools/model_manager.py install <download_id> --models-root <bundle>/models`.

---

## Environment variables

| Variable | Effect | Default |
|---|---|---|
| `AUDIOCPP_LANG` | UI language; `locales/<lang>.json` must exist | `en` |
| `AUDIOCPP_BACKEND` | Force the backend: `gpu` (=`cuda`) or `cpu` | auto-detect |
| `AUDIOCPP_BUNDLE` | Bundle root holding `cpu/ gpu/ models/` | auto-detect |
| `AUDIOCPP_SERVER` | Talk to an already-running server (`http://host:port`) instead of managing one | unset |
| `AUDIOCPP_MODEL_MANAGER` | Path to `model_manager.py` used for downloads | `<bundle>/tools/model_manager.py`, else `<repo>/tools/model_manager.py` |
| `AUDIOCPP_LOAD_TIMEOUT` | Seconds to wait for a model to finish loading | `300` |
| `AUDIOCPP_THREADS` | ggml compute threads | catalog `threads` (1); cpu backend uses cores−1 |
| `AUDIOCPP_SERVER_DEBUG` | `1` passes `--log` to the server, enabling engine `[TRACE]`/`[TIMING]` output | off |
| `AUDIOCPP_NO_BROWSER` | `1` suppresses opening a browser tab | unset |
| `AUDIOCPP_PYTHON` | Python interpreter `run_webui.sh` should use | auto-detect |
| `HF_TOKEN` | HF token for gated/private repos when downloading models (`HUGGING_FACE_HUB_TOKEN` and a cached `huggingface-cli login` also work; the UI has a token field too) | unset |

---

## FAQ

- **Port already in use**: the WebUI's managed server uses the `port` from
  `configs/models_catalog.json` (8081 as shipped) and the UI itself binds 7860. If
  something else holds the port, change `port` in the catalog, or set `AUDIOCPP_SERVER`
  to reuse an external server.
- **`model path does not exist` / not installed**: the model isn't downloaded. Use the
  model_manager command above, or download it from the WebUI.
- **Out of VRAM**: entries carry a `min_vram_gb` estimate and the UI warns when it
  exceeds the VRAM it detects. Exceeding it may still run, but it will spill into shared
  memory and slow down badly.
- **Cloned speech ends far too early (~0.4s)**: the `voice_ref` audio isn't clean, or
  `reference_text` is missing. Use a clean single-speaker reference and give it the
  matching transcript.

---

## API vs CLI performance

Same engine, same backend → **inference itself is identical**. The difference is how
model loading is amortized:

- `audiocpp_cli` **reloads the model into VRAM on every invocation** (a fixed few-second
  cost each time).
- `audiocpp_server` **loads once and stays resident**; each subsequent request costs only
  inference plus a tiny bit of transport. Local HTTP plus a few MB of wav is
  milliseconds — negligible against multi-second inference (prefer the default binary
  wav over `response_format:"json"` base64, which adds about 33%).
- The web UI adds one proxy hop compared with hitting the server directly; other programs
  talking to the server skip that hop.

**Conclusion**: going through the API adds almost nothing per generation — the one-time
warm-up is amortized by the server. Except for "generate exactly once" cases, the API is
usually **faster** than repeatedly invoking the CLI.
