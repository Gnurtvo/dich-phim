"""Lồng tiếng: tạo giọng đọc cho từng đoạn rồi ghép đúng mốc thời gian."""
import asyncio
import os
import re
import time
import wave

import numpy as np

from common import Cancelled
from ffmpeg_utils import convert_clip

SR = 24000
MAX_SPEEDUP = 1.6

_JUNK = re.compile(r"[♪♫♬♩\*_#~`<>\[\]\{\}\|\\^]")


def _clean(text):
    """Bỏ ký hiệu không đọc được, gộp khoảng trắng."""
    text = _JUNK.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


def _speakable(text):
    return any(ch.isalnum() for ch in text)


def _synth_one_edge(text, path, voice, speed, cancel):
    import edge_tts

    rate = f"{int(round((speed - 1) * 100)):+d}%"
    last = None
    for attempt in range(3):
        if cancel.is_set():
            raise Cancelled()
        try:
            asyncio.run(edge_tts.Communicate(text, voice, rate=rate).save(path))
            if os.path.exists(path) and os.path.getsize(path) > 0:
                return
            last = RuntimeError("file âm thanh rỗng")
        except Exception as e:
            last = e
        time.sleep(1.0 + attempt)
    raise last


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
                   tmp_dir, progress, cancel, log):
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
    skipped = failed = 0

    for i, seg in enumerate(segments):
        if cancel.is_set():
            raise Cancelled()
        text = _clean(seg.text)
        if not _speakable(text):
            skipped += 1          # câu chỉ có ký hiệu / rỗng: bỏ qua, không phải lỗi
            progress(int((i + 1) / len(segments) * 100))
            continue
        raw = os.path.join(tmp_dir, f"clip{i}.{'wav' if engine == 'sapi' else 'mp3'}")
        try:
            if engine == "edge":
                _synth_one_edge(text, raw, voice, speed, cancel)
            elif engine == "openai":
                _synth_one_openai(client, text, raw, voice, speed)
            else:
                sapi.save_to_file(text, raw)
                sapi.runAndWait()
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
        progress(int((i + 1) / len(segments) * 100))

    if skipped:
        log(f"Đã bỏ qua {skipped} câu không có nội dung đọc được (chỉ có ký hiệu).")
    if failed:
        log(f"Cảnh báo: {failed} câu không tạo được giọng. Xem nội dung các câu lỗi ở trên.")

    track = np.clip(track, -32768, 32767).astype(np.int16)
    with wave.open(out_wav, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SR)
        wf.writeframes(track.tobytes())
