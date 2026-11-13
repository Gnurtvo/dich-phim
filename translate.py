"""Dịch các đoạn phụ đề: Argos (offline), Google (miễn phí, cần mạng), OpenAI."""
import json
import time

from common import Cancelled, Segment

GOOGLE_CODES = {"zh": "zh-CN"}


def _copy_with(segs, texts):
    return [Segment(s.start, s.end, t) for s, t in zip(segs, texts)]


def _sleep(seconds, cancel):
    end = time.time() + seconds
    while time.time() < end:
        if cancel.is_set():
            raise Cancelled()
        time.sleep(0.1)


# ---------------- Argos (offline) ----------------
def _argos_ensure(pkg, tr, a, b, log):
    """Đảm bảo có gói a -> b. Trả về False nếu Argos không có gói này."""
    def ready():
        langs = {l.code: l for l in tr.get_installed_languages()}
        if a not in langs or b not in langs:
            return False
        return langs[a].get_translation(langs[b]) is not None

    if ready():
        return True
    log(f"Đang tải gói dịch offline {a} → {b} (chỉ lần đầu, cần mạng)...")
    pkg.update_package_index()
    found = next((p for p in pkg.get_available_packages()
                  if p.from_code == a and p.to_code == b), None)
    if found is None:
        return False
    pkg.install_from_path(found.download())
    return True


def _argos(segs, src, tgt, progress, cancel, log):
    import argostranslate.package as pkg
    import argostranslate.translate as tr

    if _argos_ensure(pkg, tr, src, tgt, log):
        route = [(src, tgt)]
    elif src != "en" and tgt != "en" and \
            _argos_ensure(pkg, tr, src, "en", log) and _argos_ensure(pkg, tr, "en", tgt, log):
        log(f"Argos không có gói trực tiếp {src} → {tgt}, dịch bắc cầu qua tiếng Anh.")
        route = [(src, "en"), ("en", tgt)]
    else:
        raise RuntimeError(
            f"Argos không hỗ trợ cặp ngôn ngữ {src} → {tgt}. Hãy chọn engine Google hoặc OpenAI."
        )

    out = []
    for i, s in enumerate(segs):
        if cancel.is_set():
            raise Cancelled()
        text = s.text
        for a, b in route:
            text = tr.translate(text, a, b)
        out.append(text)
        progress(int((i + 1) / len(segs) * 100))
    return _copy_with(segs, out)


# ---------------- Google (miễn phí) ----------------
def _google(segs, src, tgt, progress, cancel, log):
    from deep_translator import GoogleTranslator

    t = GoogleTranslator(source=GOOGLE_CODES.get(src, src) if src else "auto",
                         target=GOOGLE_CODES.get(tgt, tgt))
    out = []
    delay = 0.4          # nghỉ giữa các câu để tránh bị chặn (giới hạn ~5 yêu cầu/giây)
    failed = 0
    for i, s in enumerate(segs):
        if cancel.is_set():
            raise Cancelled()
        result = None
        for attempt in range(6):
            try:
                result = t.translate(s.text)
                break
            except Exception as e:
                wait = min(2 ** attempt * 2, 30)
                if attempt == 0:
                    log(f"Google từ chối câu {i + 1} ({str(e)[:60]}...), chờ {wait}s rồi thử lại.")
                delay = min(delay * 1.5, 2.0)   # chậm lại cho các câu sau
                _sleep(wait, cancel)
        if result is None:
            failed += 1
            log(f"✖ Không dịch được câu {i + 1}, giữ nguyên câu gốc.")
            result = s.text
        out.append(result or s.text)
        progress(int((i + 1) / len(segs) * 100))
        _sleep(delay, cancel)
    if failed:
        log(f"Cảnh báo: {failed}/{len(segs)} câu không dịch được. Thử lại sau ít phút hoặc dùng engine OpenAI.")
    return _copy_with(segs, out)


# ---------------- OpenAI ----------------
def _openai(segs, src, tgt, api_key, model, progress, cancel, log):
    from openai import OpenAI

    client = OpenAI(api_key=api_key)
    batch = 25
    out = []
    for start in range(0, len(segs), batch):
        if cancel.is_set():
            raise Cancelled()
        chunk = segs[start:start + batch]
        lines = [s.text for s in chunk]
        prompt = (
            f"Dịch các câu phụ đề sau từ '{src}' sang ngôn ngữ mã '{tgt}'. Dịch tự nhiên, ngắn gọn, "
            "phù hợp để đọc thành lời. Trả về JSON dạng {\"translations\": [...]} có đúng "
            f"{len(lines)} phần tử, cùng thứ tự.\n\n" + json.dumps(lines, ensure_ascii=False)
        )
        result = None
        for _ in range(2):
            r = client.chat.completions.create(
                model=model, response_format={"type": "json_object"},
                messages=[{"role": "user", "content": prompt}],
            )
            try:
                data = json.loads(r.choices[0].message.content)["translations"]
                if len(data) == len(lines):
                    result = [str(x) for x in data]
                    break
            except Exception:
                pass
        if result is None:
            log("OpenAI trả về sai định dạng, giữ nguyên câu gốc cho cụm này.")
            result = lines
        out.extend(result)
        progress(int(min(len(out), len(segs)) / len(segs) * 100))
    return _copy_with(segs, out)


def translate_segments(segs, src, tgt, engine, api_key, model, progress, cancel, log):
    if not segs or src == tgt:
        return segs
    if engine == "argos":
        return _argos(segs, src, tgt, progress, cancel, log)
    if engine == "openai":
        return _openai(segs, src, tgt, api_key, model, progress, cancel, log)
    return _google(segs, src, tgt, progress, cancel, log)
