"""Các kiểu dữ liệu và hằng số dùng chung."""
from dataclasses import dataclass


class Cancelled(Exception):
    """Người dùng bấm Hủy."""


@dataclass
class Segment:
    start: float
    end: float
    text: str


# (tên hiển thị, mã ngôn ngữ)
LANGS = [
    ("Tiếng Việt", "vi"),
    ("English", "en"),
    ("中文", "zh"),
    ("日本語", "ja"),
    ("한국어", "ko"),
    ("Français", "fr"),
    ("Deutsch", "de"),
    ("Español", "es"),
    ("Русский", "ru"),
    ("ภาษาไทย", "th"),
]

# Giọng Edge TTS gợi ý theo ngôn ngữ đích
EDGE_VOICES = {
    "vi": ["vi-VN-HoaiMyNeural", "vi-VN-NamMinhNeural"],
    "en": ["en-US-AriaNeural", "en-US-GuyNeural"],
    "zh": ["zh-CN-XiaoxiaoNeural", "zh-CN-YunxiNeural"],
    "ja": ["ja-JP-NanamiNeural", "ja-JP-KeitaNeural"],
    "ko": ["ko-KR-SunHiNeural", "ko-KR-InJoonNeural"],
    "fr": ["fr-FR-DeniseNeural", "fr-FR-HenriNeural"],
    "de": ["de-DE-KatjaNeural", "de-DE-ConradNeural"],
    "es": ["es-ES-ElviraNeural", "es-ES-AlvaroNeural"],
    "ru": ["ru-RU-SvetlanaNeural", "ru-RU-DmitryNeural"],
    "th": ["th-TH-PremwadeeNeural", "th-TH-NiwatNeural"],
}

OPENAI_VOICES = ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]


def _ts(seconds: float) -> str:
    ms = int(round(max(seconds, 0) * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def write_srt(segments, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for i, seg in enumerate(segments, 1):
            f.write(f"{i}\n{_ts(seg.start)} --> {_ts(seg.end)}\n{seg.text.strip()}\n\n")
