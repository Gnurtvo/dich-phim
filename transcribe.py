"""Nhận dạng giọng nói: faster-whisper (offline), OpenAI Whisper API hoặc Groq Whisper API."""
import json
import os
import time
import urllib.error
import urllib.request
import uuid

from common import Cancelled, Segment

_model_cache = {}


def transcribe_local(audio_path, model_size, language, device, progress, cancel, log):
    from faster_whisper import WhisperModel

    if device == "auto":
        try:
            import ctranslate2
            device = "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
        except Exception:
            device = "cpu"
    compute = "float16" if device == "cuda" else "int8"
    key = (model_size, device)
    if key not in _model_cache:
        log(f"Đang tải model Whisper '{model_size}' ({device}) – lần đầu sẽ tải về, hơi lâu...")
        _model_cache[key] = WhisperModel(model_size, device=device, compute_type=compute)
    model = _model_cache[key]

    lang = None if language in ("", "auto") else language
    seg_iter, info = model.transcribe(audio_path, language=lang, vad_filter=True, beam_size=5)
    detected = info.language
    log(f"Ngôn ngữ nhận dạng: {detected}")
    total = max(info.duration, 1.0)
    out = []
    for s in seg_iter:
        if cancel.is_set():
            raise Cancelled()
        text = s.text.strip()
        if text:
            out.append(Segment(s.start, s.end, text))
        progress(min(100, int(s.end / total * 100)))
    return out, detected


def transcribe_openai(audio_path, language, api_key, progress, cancel, log):
    from openai import OpenAI

    client = OpenAI(api_key=api_key)
    log("Đang gửi âm thanh lên OpenAI Whisper...")
    kwargs = {}
    if language not in ("", "auto"):
        kwargs["language"] = language
    with open(audio_path, "rb") as f:
        res = client.audio.transcriptions.create(
            model="whisper-1", file=f, response_format="verbose_json",
            timestamp_granularities=["segment"], **kwargs,
        )
    if cancel.is_set():
        raise Cancelled()

    def g(o, k):
        return o[k] if isinstance(o, dict) else getattr(o, k)

    out = [Segment(g(s, "start"), g(s, "end"), g(s, "text").strip())
           for s in (getattr(res, "segments", None) or []) if g(s, "text").strip()]
    detected = getattr(res, "language", None) or language
    progress(100)
    return out, detected


# ---------------- Groq Whisper API (nhiều key xoay vòng) ----------------
GROQ_ASR_URL = "https://api.groq.com/openai/v1/audio/transcriptions"
GROQ_ASR_MODELS = {"groq": "whisper-large-v3", "groq-turbo": "whisper-large-v3-turbo"}
CHUNK_SECONDS = 600     # cắt âm thanh 10 phút/đoạn để mỗi yêu cầu nhỏ hơn giới hạn dung lượng của Groq


def _multipart(fields, fname, data):
    b = uuid.uuid4().hex
    body = b""
    for k, v in fields:
        body += f'--{b}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode()
    body += (f'--{b}\r\nContent-Disposition: form-data; name="file"; filename="{fname}"\r\n'
             "Content-Type: audio/mpeg\r\n\r\n").encode() + data + f"\r\n--{b}--\r\n".encode()
    return body, f"multipart/form-data; boundary={b}"


def _sleep(seconds, cancel):
    end = time.time() + seconds
    while time.time() < end:
        if cancel.is_set():
            raise Cancelled()
        time.sleep(0.1)


def _groq_asr_request(path, language, model, keys, state, cancel, log):
    """Gửi một đoạn âm thanh; key hết hạn mức thì nhảy sang key khác."""
    with open(path, "rb") as f:
        data = f.read()
    fields = [("model", model), ("response_format", "verbose_json"),
              ("timestamp_granularities[]", "segment"), ("temperature", "0")]
    if language not in ("", "auto"):
        fields.append(("language", language))
    body, ctype = _multipart(fields, os.path.basename(path), data)
    limited = 0
    for attempt in range(len(keys) * 3 + 2):
        if cancel.is_set():
            raise Cancelled()
        if not keys:
            raise RuntimeError("Tất cả key Groq đều không hợp lệ.")
        key = keys[state["i"] % len(keys)]
        req = urllib.request.Request(GROQ_ASR_URL, data=body, method="POST", headers={
            "Authorization": f"Bearer {key}", "Content-Type": ctype, "User-Agent": "VideoAITool/1.0"})
        try:
            with urllib.request.urlopen(req, timeout=300) as r:
                return json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            msg = e.read().decode("utf-8", "replace")[:200]
            if e.code == 429:
                limited += 1
                log(f"Key …{key[-4:]} hết hạn mức nhận dạng, chuyển sang key khác.")
                state["i"] += 1
                if limited >= len(keys):      # đã thử hết các key: đợi một lát rồi thử lại vòng nữa
                    if limited >= len(keys) * 2:
                        raise RuntimeError("Tất cả key Groq đều hết hạn mức nhận dạng (Whisper tính theo "
                                           "số giây âm thanh mỗi giờ/ngày). Hãy đợi hoặc thêm key. " + msg)
                    log("Tất cả key đang bị giới hạn, đợi 30 giây...")
                    _sleep(30, cancel)
            elif e.code in (401, 403):
                log(f"Key …{key[-4:]} không hợp lệ, bỏ qua.")
                keys.remove(key)
            else:
                raise RuntimeError(f"Groq Whisper lỗi {e.code}: {msg}")
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            log(f"Lỗi mạng khi gọi Groq Whisper ({e}), thử lại...")
            _sleep(3, cancel)
    raise RuntimeError("Không gọi được Groq Whisper sau nhiều lần thử.")


def transcribe_groq(audio_path, language, keys, engine, progress, cancel, log, tmp_dir):
    from ffmpeg_utils import get_duration, run_ffmpeg

    keys = [k.strip() for k in keys if k and k.strip()]
    if not keys:
        raise RuntimeError("Chưa nhập API key Groq.")
    model = GROQ_ASR_MODELS.get(engine, GROQ_ASR_MODELS["groq"])
    total = get_duration(audio_path)
    chunks = [(audio_path, 0.0)]
    if total > CHUNK_SECONDS + 30:
        pat = os.path.join(tmp_dir, "chunk%03d.mp3")
        run_ffmpeg(["-i", audio_path, "-f", "segment", "-segment_time", str(CHUNK_SECONDS),
                    "-c", "copy", "-reset_timestamps", "1", pat], cancel)
        files = sorted(f for f in os.listdir(tmp_dir) if f.startswith("chunk") and f.endswith(".mp3"))
        chunks, off = [], 0.0
        for f in files:
            p = os.path.join(tmp_dir, f)
            chunks.append((p, off))
            off += get_duration(p)
    log(f"Đang gửi âm thanh lên Groq Whisper ({model}), {len(chunks)} đoạn, {len(keys)} key...")
    state = {"i": 0}
    out, detected = [], language if language not in ("", "auto") else None
    for n, (path, off) in enumerate(chunks):
        res = _groq_asr_request(path, language if detected is None else detected, model, keys, state, cancel, log)
        if detected is None and res.get("language"):
            from common import LANG_ALIASES
            low = str(res["language"]).lower()
            detected = LANG_ALIASES.get(low, low)      # chốt ngôn ngữ từ đoạn đầu cho các đoạn sau (dạng mã 2 chữ)
        for s in res.get("segments") or []:
            text = str(s.get("text", "")).strip()
            if not text or float(s.get("no_speech_prob", 0) or 0) > 0.8:
                continue      # bỏ đoạn im lặng/nhiễu mà Whisper hay "bịa" chữ
            out.append(Segment(off + float(s["start"]), off + float(s["end"]), text))
        progress(int((n + 1) / len(chunks) * 100))
    from common import LANG_ALIASES
    detected = LANG_ALIASES.get(str(detected).lower(), detected) or "auto"
    log(f"Ngôn ngữ nhận dạng: {detected}")
    return out, detected
