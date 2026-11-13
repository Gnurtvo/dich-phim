"""Dịch các đoạn phụ đề: Argos (offline), Google (miễn phí, cần mạng), OpenAI."""
import json

from common import Cancelled, Segment

GOOGLE_CODES = {"zh": "zh-CN"}


def _copy_with(segs, texts):
    return [Segment(s.start, s.end, t) for s, t in zip(segs, texts)]


def _argos(segs, src, tgt, progress, cancel, log):
    import argostranslate.package as pkg
    import argostranslate.translate as tr

    def ready():
        langs = {l.code: l for l in tr.get_installed_languages()}
        if src not in langs or tgt not in langs:
            return False
        return langs[src].get_translation(langs[tgt]) is not None

    if not ready():
        log(f"Đang tải gói dịch offline {src} → {tgt} (chỉ lần đầu, cần mạng)...")
        pkg.update_package_index()
        avail = pkg.get_available_packages()
        found = next((p for p in avail if p.from_code == src and p.to_code == tgt), None)
        if found is None:
            raise RuntimeError(
                f"Argos chưa có gói dịch trực tiếp {src} → {tgt}. Hãy chọn engine Google hoặc OpenAI."
            )
        pkg.install_from_path(found.download())
    out = []
    for i, s in enumerate(segs):
        if cancel.is_set():
            raise Cancelled()
        out.append(tr.translate(s.text, src, tgt))
        progress(int((i + 1) / len(segs) * 100))
    return _copy_with(segs, out)


def _google(segs, src, tgt, progress, cancel, log):
    from deep_translator import GoogleTranslator

    t = GoogleTranslator(source=GOOGLE_CODES.get(src, src) if src else "auto", target=GOOGLE_CODES.get(tgt, tgt))
    out = []
    for i, s in enumerate(segs):
        if cancel.is_set():
            raise Cancelled()
        try:
            out.append(t.translate(s.text) or s.text)
        except Exception as e:  # giữ nguyên câu gốc nếu lỗi
            log(f"Lỗi dịch câu {i + 1}: {e}")
            out.append(s.text)
        progress(int((i + 1) / len(segs) * 100))
    return _copy_with(segs, out)


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
