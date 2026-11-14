"""Lồng tiếng: tạo giọng đọc cho từng đoạn rồi ghép đúng mốc thời gian.

Engine: edge (Microsoft Edge TTS), fpt (FPT.AI, nhiều giọng Việt, cần API key), openai, sapi (offline
trên Windows), google (Google Dịch TTS, miễn phí, dùng làm giọng dự phòng khi engine chính lỗi).
"""
import asyncio
import json
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

FPT_URL = "https://api.fpt.ai/hmi/tts/v5"
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


# ---------------- Edge TTS ----------------
def _parse_edge_voice(spec):
    """'vi-VN-HoaiMyNeural;pitch=+10Hz;rate=-8' -> (tên giọng, pitch, độ lệch tốc độ %)."""
    parts = [p.strip() for p in (spec or "").split(";") if p.strip()]
    name = parts[0] if parts else ""
    pitch, rate_off = "+0Hz", 0
    for p in parts[1:]:
        k, _, v = p.partition("=")
        try:
            if k == "pitch":
                pitch = f"{int(float(v.replace('Hz', ''))):+d}Hz"
            elif k == "rate":
                rate_off = int(float(v.replace("%", "")))
        except ValueError:
            pass
    return name, pitch, rate_off


def _synth_one_edge(text, path, spec, speed, cancel):
    import edge_tts

    name, pitch, rate_off = _parse_edge_voice(spec)
    rate = f"{int(round((speed - 1) * 100)) + rate_off:+d}%"
    last = None
    for attempt in range(3):
        if cancel.is_set():
            raise Cancelled()
        t = text
        if attempt >= 1 and not re.search(r"[.!?…。！？]$", t):
            t += "."          # lần thử lại: đổi nhẹ nội dung (lỗi "không có âm thanh" đôi khi chỉ xảy ra với một chuỗi cụ thể)
        try:
            if os.path.exists(path):
                os.remove(path)
            asyncio.run(edge_tts.Communicate(t, name, rate=rate, pitch=pitch).save(path))
            if os.path.exists(path) and os.path.getsize(path) > 0:
                time.sleep(0.15)   # giãn nhịp yêu cầu để tránh bị dịch vụ chặn tạm thời
                return
            last = RuntimeError("file âm thanh rỗng")
        except Exception as e:
            last = e
        _sleep(1.0 + attempt, cancel)
    raise last


# ---------------- FPT.AI ----------------
def _synth_one_fpt(text, path, voice, speed, api_key, cancel):
    if len(text) < 3:
        text += "..."          # FPT yêu cầu tối thiểu 3 ký tự
    sp = max(-3, min(3, round((speed - 1) * 5)))
    req = urllib.request.Request(
        FPT_URL, data=text.encode("utf-8"), method="POST",
        headers={"api_key": api_key, "voice": voice or "banmai", "speed": str(sp), "format": "mp3"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            resp = json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"FPT.AI lỗi {e.code}: {e.read().decode('utf-8', 'replace')[:150]}")
    if resp.get("error") != 0 or not resp.get("async"):
        raise RuntimeError(f"FPT.AI báo lỗi: {resp.get('message', resp)}")
    url = resp["async"]
    deadline = time.time() + 120
    while time.time() < deadline:     # file được tạo bất đồng bộ (vài giây đến vài chục giây)
        _sleep(1.5, cancel)
        try:
            with urllib.request.urlopen(urllib.request.Request(url), timeout=30) as r:
                data = r.read()
            if data:
                with open(path, "wb") as f:
                    f.write(data)
                return
        except urllib.error.HTTPError as e:
            if e.code not in (403, 404, 409, 425):
                raise RuntimeError(f"FPT.AI tải file lỗi {e.code}")
        except urllib.error.URLError:
            pass
    raise RuntimeError("FPT.AI quá thời gian chờ tạo giọng")


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


# ---------------- OpenAI / SAPI ----------------
def _make_sapi():
    import pyttsx3
    return pyttsx3.init()


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
    sapi = None
    client = None
    if engine == "sapi":
        sapi = _make_sapi()
        if voice:
            for v in sapi.getProperty("voices"):
                if voice.lower() in (v.name + v.id).lower():
                    sapi.setProperty("voice", v.id)
                    break
            else:
                log(f"Không thấy giọng SAPI chứa '{voice}', dùng giọng mặc định.")
    elif engine == "openai":
        from openai import OpenAI
        client = OpenAI(api_key=api_key)

    last_end = max((s.end for s in segments), default=0)
    length = int((max(total_duration, last_end) + 2) * SR)
    track = np.zeros(length, dtype=np.int32)
    skipped = failed = fallback = 0

    def raw_path(i):
        return os.path.join(tmp_dir, f"clip{i}.{'wav' if engine == 'sapi' else 'mp3'}")

    # FPT tạo file bất đồng bộ nên gửi song song nhiều câu cho nhanh
    prefetch_err = {}
    main_base, main_span = 0, 100
    if engine == "fpt":
        from concurrent.futures import ThreadPoolExecutor

        jobs = [(i, _clean(s.text)) for i, s in enumerate(segments)]
        jobs = [(i, t) for i, t in jobs if _speakable(t)]

        def work(job):
            i, t = job
            try:
                _synth_one_fpt(t, raw_path(i), voice, speed, api_key, cancel)
            except Exception as e:
                prefetch_err[i] = e

        done = 0
        with ThreadPoolExecutor(max_workers=4) as pool:
            for _ in pool.map(work, jobs):
                done += 1
                progress(int(done / max(len(jobs), 1) * 50))
        if cancel.is_set():
            raise Cancelled()
        main_base, main_span = 50, 50

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
                if engine == "edge":
                    _synth_one_edge(text, raw, voice, speed, cancel)
                elif engine == "fpt":
                    if i in prefetch_err:
                        raise prefetch_err[i]
                elif engine == "openai":
                    _synth_one_openai(client, text, raw, voice, speed)
                elif engine == "google":
                    _synth_one_google(text, raw, lang)
                else:
                    sapi.save_to_file(text, raw)
                    sapi.runAndWait()
            except Cancelled:
                raise
            except Exception as e:
                if engine not in ("edge", "fpt", "openai"):
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
