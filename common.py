"""Các kiểu dữ liệu và hằng số dùng chung."""
import re
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

# Giọng Edge tiếng Việt kèm tinh chỉnh: "tên giọng;pitch=<Hz>;rate=<% tốc độ>".
# Edge chỉ có 2 giọng Việt; các bản 'ngọt ngào', 'trầm ấm'... là chỉnh cao độ/tốc độ của 2 giọng đó.
EDGE_VI_PRESETS = [
    ("Hoài My – chuẩn (nữ)", "vi-VN-HoaiMyNeural"),
    ("Hoài My – ngọt ngào, nhẹ nhàng (nữ)", "vi-VN-HoaiMyNeural;pitch=+10Hz;rate=-8"),
    ("Hoài My – tươi vui, trẻ trung (nữ)", "vi-VN-HoaiMyNeural;pitch=+18Hz;rate=+6"),
    ("Hoài My – chậm rãi, kể chuyện (nữ)", "vi-VN-HoaiMyNeural;pitch=-4Hz;rate=-15"),
    ("Nam Minh – chuẩn (nam)", "vi-VN-NamMinhNeural"),
    ("Nam Minh – trầm ấm (nam)", "vi-VN-NamMinhNeural;pitch=-10Hz;rate=-6"),
]

# Giọng FPT.AI (tiếng Việt)
FPT_VOICES = [
    ("Ban Mai – nữ miền Bắc", "banmai"),
    ("Thu Minh – nữ miền Bắc", "thuminh"),
    ("Lan Nhi – nữ miền Nam", "lannhi"),
    ("Linh San – nữ miền Nam", "linhsan"),
    ("Mỹ An – nữ miền Trung", "myan"),
    ("Lê Minh – nam miền Bắc", "leminh"),
    ("Gia Huy – nam miền Trung", "giahuy"),
]


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


_TIME_RE = re.compile(
    r"(\d+):(\d{1,2}):(\d{1,2})[,.](\d{1,3})\s*-->\s*(\d+):(\d{1,2}):(\d{1,2})[,.](\d{1,3})")


def _secs(h, m, s, ms) -> float:
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms.ljust(3, "0")) / 1000


def read_srt(path: str):
    """Đọc file .srt (chịu được BOM, CRLF, đánh số sai, nhiều dòng, thẻ HTML/ASS)."""
    with open(path, encoding="utf-8-sig", errors="replace") as f:
        raw = f.read().replace("\r\n", "\n").replace("\r", "\n")
    segs = []
    for block in re.split(r"\n\s*\n", raw.strip()):
        lines = block.split("\n")
        for idx, line in enumerate(lines):
            m = _TIME_RE.search(line)
            if not m:
                continue
            g = m.groups()
            text = " ".join(x.strip() for x in lines[idx + 1:] if x.strip())
            text = re.sub(r"<[^>]+>|\{[^}]*\}", "", text).strip()
            if text:
                segs.append(Segment(_secs(*g[:4]), _secs(*g[4:]), text))
            break
    return segs
