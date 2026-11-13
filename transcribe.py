"""Nhận dạng giọng nói: faster-whisper (offline) hoặc OpenAI Whisper API."""
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
