"""Quy trình xử lý một video: nhận dạng → dịch → lồng tiếng → ghép/xóa metadata."""
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

from common import Cancelled, write_srt
from ffmpeg_utils import extract_audio, final_mux, get_duration, has_audio


@dataclass
class Options:
    output_dir: str = ""
    # các bước
    do_transcribe: bool = True      # xuất phụ đề gốc
    do_translate: bool = True
    do_dub: bool = True
    embed_subs: bool = False
    strip_metadata: bool = True
    # nhận dạng
    asr_engine: str = "local"       # local | openai
    whisper_model: str = "small"
    device: str = "auto"
    src_lang: str = "auto"
    # dịch
    tr_engine: str = "google"       # argos | google | openai
    tgt_lang: str = "vi"
    openai_model: str = "gpt-4o-mini"
    # lồng tiếng
    tts_engine: str = "edge"        # edge | sapi | openai
    voice: str = ""
    tts_speed: float = 1.0
    orig_volume: float = 0.1
    openai_key: str = ""


def process_video(video: str, o: Options, log, progress, cancel):
    """progress(percent 0-100) cho riêng video này."""
    from transcribe import transcribe_local, transcribe_openai
    from translate import translate_segments
    from tts import synthesize_dub

    def check():
        if cancel.is_set():
            raise Cancelled()

    stem = Path(video).stem
    out_dir = Path(o.output_dir or Path(video).parent)
    out_dir.mkdir(parents=True, exist_ok=True)

    needs_asr = o.do_transcribe or o.do_translate or o.do_dub or o.embed_subs
    needs_video = o.do_dub or o.embed_subs or o.strip_metadata
    if needs_asr and not has_audio(video):
        raise RuntimeError("Video không có luồng âm thanh để nhận dạng.")
    if "openai" in (o.tr_engine if o.do_translate else "", o.tts_engine if o.do_dub else "",
                    o.asr_engine if needs_asr else "") and not o.openai_key:
        raise RuntimeError("Bạn đã chọn tính năng dùng OpenAI nhưng chưa nhập API key.")

    segments = []
    src_srt = tgt_srt = None
    detected = o.src_lang

    with tempfile.TemporaryDirectory(prefix="vat_") as tmp:
        if needs_asr:
            log("▶ Trích âm thanh...")
            ext = "mp3" if o.asr_engine == "openai" else "wav"
            audio = os.path.join(tmp, f"audio.{ext}")
            extract_audio(video, audio, cancel)
            progress(5)
            check()

            log("▶ Nhận dạng giọng nói...")
            p = lambda v: progress(5 + int(v * 0.35))
            if o.asr_engine == "openai":
                segments, detected = transcribe_openai(audio, o.src_lang, o.openai_key, p, cancel, log)
            else:
                segments, detected = transcribe_local(audio, o.whisper_model, o.src_lang,
                                                      o.device, p, cancel, log)
            if not segments:
                raise RuntimeError("Không nhận dạng được lời thoại nào trong video.")
            log(f"  Nhận dạng được {len(segments)} đoạn.")
            src_srt = str(out_dir / f"{stem}.{detected}.srt")
            write_srt(segments, src_srt)
            log(f"  Đã lưu phụ đề gốc: {src_srt}")

        final_segments = segments
        if o.do_translate and segments:
            log("▶ Dịch phụ đề...")
            final_segments = translate_segments(
                segments, detected, o.tgt_lang, o.tr_engine, o.openai_key, o.openai_model,
                lambda v: progress(40 + int(v * 0.15)), cancel, log)
            tgt_srt = str(out_dir / f"{stem}.{o.tgt_lang}.srt")
            write_srt(final_segments, tgt_srt)
            log(f"  Đã lưu phụ đề dịch: {tgt_srt}")
        progress(55)
        check()

        dub_wav = None
        if o.do_dub and final_segments:
            log("▶ Tạo giọng lồng tiếng...")
            dub_wav = os.path.join(tmp, "dub.wav")
            synthesize_dub(final_segments, o.tts_engine, o.voice, o.tts_speed, o.openai_key,
                           get_duration(video), dub_wav, tmp,
                           lambda v: progress(55 + int(v * 0.30)), cancel, log)
        progress(85)
        check()

        if needs_video:
            out_video = str(out_dir / f"{stem}_output.mp4")
            log("▶ Xuất video cuối...")
            final_mux(video, out_video, dub_wav=dub_wav,
                      srt=(tgt_srt or src_srt) if o.embed_subs else None,
                      orig_volume=o.orig_volume, strip_metadata=o.strip_metadata, cancel=cancel)
            log(f"✔ Hoàn tất: {out_video}")
        else:
            log("✔ Hoàn tất (chỉ xuất phụ đề).")
        progress(100)
