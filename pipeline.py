"""Quy trình xử lý một video.

Có 3 chế độ:
- full         : nhận dạng → dịch → lồng tiếng → ghép/xóa metadata
- extract_only : chỉ nhận dạng và lưu .srt gốc rồi dừng (để bạn tự dịch)
- from_srt     : dùng file .srt đã dịch → lồng tiếng → ghép/xóa metadata
"""
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

from common import Cancelled, read_srt, write_srt
from ffmpeg_utils import extract_audio, final_mux, get_duration, has_audio


@dataclass
class Options:
    output_dir: str = ""
    mode: str = "full"              # full | extract_only | from_srt
    srt_file: str = ""              # file .srt đã dịch (chế độ from_srt); rỗng = tự tìm
    # các bước
    do_transcribe: bool = True      # xuất phụ đề gốc
    do_translate: bool = True
    do_dub: bool = True
    embed_subs: bool = False        # nhúng phụ đề mềm
    burn_subs: bool = False         # ghi cứng phụ đề vào hình (encode lại)
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


def find_translated_srt(video, out_dir, tgt):
    """Tìm <tên video>.<ngôn ngữ đích>.srt (hoặc <tên video>.srt) trong thư mục xuất và thư mục video."""
    stem = Path(video).stem
    for d in (Path(out_dir), Path(video).parent):
        for name in (f"{stem}.{tgt}.srt", f"{stem}.srt"):
            p = d / name
            if p.is_file():
                return str(p)
    return None


def process_video(video: str, o: Options, log, progress, cancel):
    """progress(percent 0-100) cho riêng video này."""
    def check():
        if cancel.is_set():
            raise Cancelled()

    mode = o.mode
    stem = Path(video).stem
    out_dir = Path(o.output_dir or Path(video).parent)
    out_dir.mkdir(parents=True, exist_ok=True)

    if mode == "extract_only":
        needs_asr = True
    elif mode == "from_srt":
        needs_asr = False
    else:
        needs_asr = o.do_transcribe or o.do_translate or o.do_dub or o.embed_subs or o.burn_subs
    needs_video = mode != "extract_only" and (o.do_dub or o.embed_subs or o.burn_subs or o.strip_metadata)

    if needs_asr and not has_audio(video):
        raise RuntimeError("Video không có luồng âm thanh để nhận dạng.")

    used = []
    if needs_asr:
        used.append(o.asr_engine)
    if mode == "full" and o.do_translate:
        used.append(o.tr_engine)
    if mode in ("full", "from_srt") and o.do_dub:
        used.append(o.tts_engine)
    if "openai" in used and not o.openai_key:
        raise RuntimeError("Bạn đã chọn tính năng dùng OpenAI nhưng chưa nhập API key.")

    segments = []
    src_srt = tgt_srt = None
    detected = o.src_lang

    with tempfile.TemporaryDirectory(prefix="vat_") as tmp:
        # ---------- nhận dạng ----------
        if needs_asr:
            from transcribe import transcribe_local, transcribe_openai

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

            if mode == "extract_only":
                log("⏸ Đã dừng sau khi tách phụ đề.")
                log(f"  Hãy dịch file trên, lưu thành '{stem}.{o.tgt_lang}.srt' (giữ nguyên mốc thời gian),")
                log("  rồi chọn chế độ 'Bước 2' để tiếp tục lồng tiếng.")
                progress(100)
                return

        # ---------- dịch / đọc phụ đề đã dịch ----------
        final_segments = segments
        if mode == "from_srt":
            srt_path = o.srt_file or find_translated_srt(video, out_dir, o.tgt_lang)
            if not srt_path or not os.path.isfile(srt_path):
                raise RuntimeError(
                    f"Không tìm thấy phụ đề đã dịch. Hãy đặt tên '{stem}.{o.tgt_lang}.srt' cạnh video "
                    "hoặc trong thư mục xuất, hoặc chọn file .srt thủ công.")
            log(f"▶ Đọc phụ đề đã dịch: {srt_path}")
            final_segments = read_srt(srt_path)
            if not final_segments:
                raise RuntimeError("File .srt trống hoặc sai định dạng.")
            log(f"  Đọc được {len(final_segments)} đoạn.")
            tgt_srt = srt_path
        elif o.do_translate and segments:
            from translate import translate_segments

            log("▶ Dịch phụ đề...")
            final_segments = translate_segments(
                segments, detected, o.tgt_lang, o.tr_engine, o.openai_key, o.openai_model,
                lambda v: progress(40 + int(v * 0.15)), cancel, log)
            tgt_srt = str(out_dir / f"{stem}.{o.tgt_lang}.srt")
            write_srt(final_segments, tgt_srt)
            log(f"  Đã lưu phụ đề dịch: {tgt_srt}")
        progress(55)
        check()

        # ---------- lồng tiếng ----------
        dub_wav = None
        if o.do_dub and final_segments:
            from tts import synthesize_dub

            log("▶ Tạo giọng lồng tiếng...")
            dub_wav = os.path.join(tmp, "dub.wav")
            synthesize_dub(final_segments, o.tts_engine, o.voice, o.tts_speed, o.openai_key,
                           get_duration(video), dub_wav, tmp,
                           lambda v: progress(55 + int(v * 0.30)), cancel, log)
        progress(85)
        check()

        # ---------- ghép video ----------
        sub_file = tgt_srt or src_srt
        if needs_video:
            out_video = str(out_dir / f"{stem}_output.mp4")
            log("▶ Xuất video cuối..." + (" (ghi cứng phụ đề nên sẽ lâu hơn)" if o.burn_subs and sub_file else ""))
            final_mux(video, out_video, dub_wav=dub_wav,
                      srt=sub_file if (o.embed_subs and not o.burn_subs) else None,
                      burn_srt=sub_file if o.burn_subs else None,
                      orig_volume=o.orig_volume, strip_metadata=o.strip_metadata, cancel=cancel)
            log(f"✔ Hoàn tất: {out_video}")
        else:
            log("✔ Hoàn tất (chỉ xuất phụ đề).")
        progress(100)
