"""Lồng tiếng: tạo giọng đọc cho từng đoạn rồi ghép đúng mốc thời gian.

Engine: vieneu (VieNeu-TTS, offline, tiếng Việt), openai, google (Google Dịch TTS, miễn phí, cũng là giọng
dự phòng khi engine chính lỗi ở một câu).
"""
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
import wave

import numpy as np

from common import Cancelled
from ffmpeg_utils import convert_clip

SR = 24000
MAX_SPEEDUP = 1.6

GOOGLE_TTS_URL = "https://translate.google.com/translate_tts"
GOOGLE_LANG = {"zh": "zh-CN"}
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
      "Chrome/124.0 Safari/537.36")

_JUNK = re.compile(r"[♪♫♬♩\*_#~`<>\[\]\{\}\|\\^]")


def _clean(text):
    """Bỏ ký hiệu không đọc được, gộp khoảng trắng."""
    text = _JUNK.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


def _speakable(text):
    return any(ch.isalnum() for ch in text)


def _sleep(seconds, cancel):
    end = time.time() + seconds
    while time.time() < end:
        if cancel.is_set():
            raise Cancelled()
        time.sleep(0.1)


def _split_text(text, n=180):
    """Chia câu dài thành các đoạn ≤ n ký tự (Google TTS giới hạn độ dài mỗi yêu cầu)."""
    if len(text) <= n:
        return [text]
    chunks, cur = [], ""
    for piece in re.split(r"(?<=[,.;:!?…，。！？])\s*", text):
        words = [piece] if len(piece) <= n else piece.split(" ")
        for w in words:
            if cur and len(cur) + len(w) + 1 > n:
                chunks.append(cur)
                cur = w
            else:
                cur = f"{cur} {w}".strip() if cur else w
    if cur:
        chunks.append(cur)
    return chunks


# ---------------- Google TTS (miễn phí, dùng làm dự phòng) ----------------
def _synth_one_google(text, path, lang):
    data = b""
    for chunk in _split_text(text):
        q = urllib.parse.urlencode({"ie": "UTF-8", "q": chunk, "tl": GOOGLE_LANG.get(lang, lang),
                                    "client": "tw-ob"})
        req = urllib.request.Request(f"{GOOGLE_TTS_URL}?{q}",
                                     headers={"User-Agent": UA, "Referer": "https://translate.google.com/"})
        with urllib.request.urlopen(req, timeout=30) as r:
            part = r.read()
        if not part:
            raise RuntimeError("Google TTS trả về dữ liệu rỗng")
        data += part
    with open(path, "wb") as f:
        f.write(data)


# ---------------- VieNeu-TTS (offline, tiếng Việt) ----------------
_VIENEU = {}


def _vieneu_engine(log):
    if "tts" not in _VIENEU:
        try:
            from vieneu import Vieneu
        except Exception as e:
            raise RuntimeError(f"Chưa có thư viện VieNeu-TTS trong bản cài này ({e}). "
                               "Hãy build lại với 'pip install vieneu' hoặc chọn giọng khác.")
        log("Đang nạp VieNeu-TTS (lần đầu sẽ tải model từ Hugging Face, cần mạng và vài phút)...")
        try:
            _VIENEU["tts"] = Vieneu()
        except Exception as e:
            import traceback
            log("Chi tiết lỗi nạp VieNeu-TTS:\n" + traceback.format_exc()[-900:])
            raise RuntimeError(f"Không nạp được VieNeu-TTS: {e}")
    return _VIENEU["tts"]


def _vieneu_voice(engine, name, log):
    """Tìm giọng preset theo mã hoặc một phần tên; không thấy thì dùng giọng mặc định."""
    if not name:
        return None
    try:
        voices = [(str(l), str(v)) for l, v in engine.list_preset_voices()]
    except Exception:
        voices = []
    low = name.lower()
    for test in (lambda l, v: v.lower() == low,                     # mã giọng chính xác
                 lambda l, v: l.lower() == low,                     # nhãn chính xác
                 lambda l, v: low in (l + " " + v).lower()):        # một phần tên
        for label, vid in voices:
            if test(label, vid):
                return vid
    log(f"Không thấy giọng VieNeu chứa '{name}', dùng giọng mặc định.")
    return None


_VN_TRACE = []


def list_vieneu_voices(log):
    """Danh sách giọng có sẵn [(nhãn, mã)] – cần nạp model nên chạy ở luồng nền."""
    eng = _vieneu_engine(log)
    return [(str(l), str(v)) for l, v in eng.list_preset_voices()]


def _synth_one_vieneu(engine, text, path, voice_id, log=None):
    kw = {"voice": voice_id} if voice_id else {}
    try:
        audio = engine.infer(text=text, **kw)
        engine.save(audio, path)
    except Exception:
        if log and not _VN_TRACE:      # chỉ in chi tiết một lần để dễ chẩn đoán
            import traceback
            _VN_TRACE.append(1)
            log("Chi tiết lỗi VieNeu-TTS (gửi cho người hỗ trợ):\n" + traceback.format_exc()[-1500:])
        raise
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        raise RuntimeError("VieNeu-TTS không tạo ra âm thanh")


# ---------------- OpenAI / SAPI ----------------
def _synth_one_openai(client, text, path, voice, speed):
    with client.audio.speech.with_streaming_response.create(
        model="tts-1", voice=voice or "alloy", input=text, speed=max(0.25, min(4.0, speed))
    ) as r:
        r.stream_to_file(path)


def _read_wav(path):
    with wave.open(path, "rb") as wf:
        n = wf.getnframes()
        return np.frombuffer(wf.readframes(n), dtype=np.int16), wf.getframerate()


def synthesize_dub(segments, engine, voice, speed, api_key, total_duration, out_wav,
                   tmp_dir, progress, cancel, log, lang="vi"):
    """Tạo file WAV mono 24 kHz dài bằng video, mỗi câu đặt tại mốc thời gian gốc."""
    client = None
    if engine == "openai":
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
    vn = vn_voice = None
    if engine == "vieneu":
        vn = _vieneu_engine(log)
        vn_voice = _vieneu_voice(vn, voice, log)

    last_end = max((s.end for s in segments), default=0)
    length = int((max(total_duration, last_end) + 2) * SR)
    track = np.zeros(length, dtype=np.int32)
    skipped = failed = fallback = 0

    def raw_path(i):
        return os.path.join(tmp_dir, f"raw{i}.{'wav' if engine == 'vieneu' else 'mp3'}")

    main_base, main_span = 0, 100

    for i, seg in enumerate(segments):
        if cancel.is_set():
            raise Cancelled()
        text = _clean(seg.text)
        if not _speakable(text):
            skipped += 1          # câu chỉ có ký hiệu / rỗng: bỏ qua, không phải lỗi
            progress(main_base + int((i + 1) / len(segments) * main_span))
            continue
        raw = raw_path(i)
        try:
            try:
                if engine == "openai":
                    _synth_one_openai(client, text, raw, voice, speed)
                elif engine == "vieneu":
                    _synth_one_vieneu(vn, text, raw, vn_voice, log)
                elif engine == "google":
                    _synth_one_google(text, raw, lang)
                else:
                    raise RuntimeError(f"Engine TTS không hỗ trợ: {engine}")
            except Cancelled:
                raise
            except Exception as e:
                if engine not in ("openai", "vieneu"):
                    raise
                # engine chính lỗi: dùng giọng Google dự phòng để câu này không bị im lặng
                log(f"Câu {i + 1}: {engine} lỗi ({str(e)[:80]}) → dùng giọng Google dự phòng.")
                _synth_one_google(text, raw, lang)
                fallback += 1

            wav = os.path.join(tmp_dir, f"clip{i}.wav")
            convert_clip(raw, wav, 1.0, cancel)
            data, _ = _read_wav(wav)

            # khoảng thời gian cho phép: tới đầu câu kế tiếp (tối đa dư 1 giây sau seg.end)
            nxt = segments[i + 1].start if i + 1 < len(segments) else seg.end + 1.0
            slot = max(seg.end, min(nxt - 0.05, seg.end + 1.0)) - seg.start
            dur = len(data) / SR
            if slot > 0.3 and dur > slot:
                factor = min(dur / slot, MAX_SPEEDUP)
                wav2 = os.path.join(tmp_dir, f"clip{i}_fast.wav")
                convert_clip(raw, wav2, factor, cancel)
                data, _ = _read_wav(wav2)

            pos = int(seg.start * SR)
            end = min(pos + len(data), length)
            track[pos:end] += data[: end - pos]
        except Cancelled:
            raise
        except Exception as e:
            failed += 1
            log(f"Lỗi tạo giọng câu {i + 1}: {e} | nội dung: \"{text[:60]}\"")
        progress(main_base + int((i + 1) / len(segments) * main_span))

    if skipped:
        log(f"Đã bỏ qua {skipped} câu không có nội dung đọc được (chỉ có ký hiệu).")
    if fallback:
        log(f"Đã dùng giọng Google dự phòng cho {fallback} câu (giọng sẽ khác giọng chính).")
    if failed:
        log(f"Cảnh báo: {failed} câu không tạo được giọng. Xem nội dung các câu lỗi ở trên.")

    track = np.clip(track, -32768, 32767).astype(np.int16)
    with wave.open(out_wav, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SR)
        wf.writeframes(track.tobytes())
