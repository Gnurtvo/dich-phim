"""Bọc các lệnh FFmpeg/FFprobe."""
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from common import Cancelled

CREATE_NO_WINDOW = 0x08000000 if os.name == "nt" else 0


class FFmpegError(RuntimeError):
    pass


def _search_dirs():
    if getattr(sys, "frozen", False):
        yield Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent))
        yield Path(sys.executable).parent
    else:
        yield Path(__file__).resolve().parent


def find_tool(name: str) -> str:
    exe = name + (".exe" if os.name == "nt" else "")
    for d in _search_dirs():
        for cand in (d / "ffmpeg" / exe, d / exe):
            if cand.exists():
                return str(cand)
    found = shutil.which(name)
    if found:
        return found
    raise FFmpegError(
        f"Không tìm thấy {name}. Hãy cài FFmpeg và thêm vào PATH, "
        f"hoặc đặt {exe} vào thư mục 'ffmpeg' cạnh phần mềm."
    )


def run_ffmpeg(args, cancel=None, cwd=None):
    cmd = [find_tool("ffmpeg"), "-y", "-hide_banner", "-loglevel", "error", "-nostdin", *map(str, args)]
    with tempfile.TemporaryFile() as err:
        proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=err,
                                creationflags=CREATE_NO_WINDOW, cwd=cwd)
        while True:
            try:
                proc.wait(timeout=0.3)
                break
            except subprocess.TimeoutExpired:
                if cancel is not None and cancel.is_set():
                    proc.kill()
                    proc.wait()
                    raise Cancelled()
        if proc.returncode != 0:
            err.seek(0)
            msg = err.read().decode("utf-8", "replace").strip()
            raise FFmpegError(msg or f"ffmpeg lỗi (mã {proc.returncode})")


def get_duration(path: str) -> float:
    out = subprocess.run(
        [find_tool("ffprobe"), "-v", "error", "-print_format", "json", "-show_format", str(path)],
        capture_output=True, creationflags=CREATE_NO_WINDOW,
    )
    if out.returncode != 0:
        raise FFmpegError(out.stderr.decode("utf-8", "replace"))
    return float(json.loads(out.stdout)["format"].get("duration", 0) or 0)


def has_audio(path: str) -> bool:
    out = subprocess.run(
        [find_tool("ffprobe"), "-v", "error", "-select_streams", "a", "-show_entries",
         "stream=index", "-print_format", "json", str(path)],
        capture_output=True, creationflags=CREATE_NO_WINDOW,
    )
    try:
        return bool(json.loads(out.stdout).get("streams"))
    except Exception:
        return False


def extract_audio(video: str, out_path: str, cancel=None):
    """Trích âm thanh mono 16 kHz. Đuôi .mp3 -> nén nhỏ (dùng cho API), .wav -> không nén."""
    args = ["-i", video, "-vn", "-ac", "1", "-ar", "16000"]
    if out_path.lower().endswith(".mp3"):
        args += ["-b:a", "48k"]
    run_ffmpeg([*args, out_path], cancel)


def convert_clip(src: str, dst: str, tempo: float = 1.0, cancel=None):
    """Chuyển clip TTS sang WAV mono 24 kHz, có thể tăng tốc bằng atempo (1.0 - 2.0)."""
    args = ["-i", src, "-ac", "1", "-ar", "24000"]
    if tempo > 1.01:
        args += ["-filter:a", f"atempo={min(tempo, 2.0):.3f}"]
    run_ffmpeg([*args, "-c:a", "pcm_s16le", dst], cancel)


def _effective_font_size(video, size):
    """Cỡ chữ tính theo cạnh NGẮN của video.

    ASS/libass quy cỡ chữ theo chiều cao khung hình, nên video dọc (9:16) sẽ bị chữ to quá
    gấp ~1,8 lần. Hàm này thu nhỏ lại để cùng một con số cho chữ cùng kích thước ở cả hai loại video.
    """
    try:
        out = subprocess.run(
            [find_tool("ffprobe"), "-v", "error", "-select_streams", "v:0", "-show_entries",
             "stream=width,height", "-print_format", "json", str(video)],
            capture_output=True, creationflags=CREATE_NO_WINDOW)
        st = json.loads(out.stdout)["streams"][0]
        w, h = int(st["width"]), int(st["height"])
        return max(4, round(int(size) * min(w, h) / h))
    except Exception:
        return int(size)


def final_mux(video, out_path, dub_wav=None, srt=None, orig_volume=0.1, strip_metadata=False,
              burn_srt=None, sub_size=14, cancel=None):
    """Ghép video + (tiếng lồng) + (phụ đề mềm hoặc ghi cứng), tùy chọn xóa metadata."""
    video, out_path = os.path.abspath(video), os.path.abspath(out_path)
    dub_wav = os.path.abspath(dub_wav) if dub_wav else None
    srt = os.path.abspath(srt) if srt and not burn_srt else None
    sub_dir = None
    if burn_srt:
        # ffmpeg chạy trong thư mục tạm với tên file đơn giản để tránh lỗi đường dẫn Windows (C:\...)
        sub_dir = tempfile.mkdtemp(prefix="vat_sub_")
        shutil.copyfile(burn_srt, os.path.join(sub_dir, "sub.srt"))
    args = ["-i", video]
    idx = 1
    dub_idx = srt_idx = None
    if dub_wav:
        args += ["-i", dub_wav]
        dub_idx = idx
        idx += 1
    if srt:
        args += ["-i", srt]
        srt_idx = idx

    args += ["-map", "0:v:0"]
    if dub_idx is not None:
        if orig_volume > 0 and has_audio(video):
            fc = (f"[0:a:0]volume={orig_volume:.2f}[o];[{dub_idx}:a]volume=1.0[d];"
                  f"[o][d]amix=inputs=2:duration=longest:normalize=0[a]")
            args += ["-filter_complex", fc, "-map", "[a]"]
        else:
            args += ["-map", f"{dub_idx}:a"]
        args += ["-c:a", "aac", "-b:a", "192k"]
    else:
        args += ["-map", "0:a?", "-c:a", "copy"]
    if burn_srt:
        style = f"FontName=Arial,FontSize={_effective_font_size(video, sub_size)},Outline=1"
        args += ["-vf", f"subtitles=sub.srt:force_style='{style}'",
                 "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p"]
    else:
        args += ["-c:v", "copy"]

    if srt_idx is not None:
        sub_codec = "srt" if out_path.lower().endswith(".mkv") else "mov_text"
        args += ["-map", f"{srt_idx}:0", "-c:s", sub_codec]

    if strip_metadata:
        args += [
            "-map_metadata", "-1", "-map_chapters", "-1",
            "-fflags", "+bitexact", "-flags:v", "+bitexact", "-flags:a", "+bitexact",
            "-metadata:s:v:0", "handler_name=", "-metadata:s:a:0", "handler_name=",
        ]
    args += ["-shortest"] if dub_idx is None else []
    try:
        run_ffmpeg([*args, out_path], cancel, cwd=sub_dir)
    finally:
        if sub_dir:
            shutil.rmtree(sub_dir, ignore_errors=True)
