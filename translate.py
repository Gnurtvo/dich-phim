"""Dịch các đoạn phụ đề: Groq (nhiều key xoay vòng) / Gemini / Ollama (offline) / OpenAI (hiểu ngữ cảnh), Google (dự phòng)."""
import json
import re
import time
import urllib.error
import urllib.request

from common import LANGS, Cancelled, Segment

GOOGLE_CODES = {"zh": "zh-CN"}
LANG_NAMES = {code: name for name, code in LANGS}


def _copy_with(segs, texts):
    return [Segment(s.start, s.end, t) for s, t in zip(segs, texts)]


def _sleep(seconds, cancel):
    end = time.time() + seconds
    while time.time() < end:
        if cancel.is_set():
            raise Cancelled()
        time.sleep(0.1)


# ---------------- Google Dịch (miễn phí, từng câu) ----------------
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
                delay = min(delay * 1.5, 2.0)
                _sleep(wait, cancel)
        if result is None:
            failed += 1
            log(f"✖ Không dịch được câu {i + 1}, giữ nguyên câu gốc.")
            result = s.text
        out.append(result or s.text)
        progress(int((i + 1) / len(segs) * 100))
        _sleep(delay, cancel)
    if failed:
        log(f"Cảnh báo: {failed}/{len(segs)} câu không dịch được. Thử lại sau ít phút hoặc dùng Gemini.")
    return _copy_with(segs, out)


# ---------------- Dịch bằng LLM, có ngữ cảnh ----------------
def _system_prompt(src, tgt, note):
    s, t = LANG_NAMES.get(src, src), LANG_NAMES.get(tgt, tgt)
    sys = (
        f"Bạn là biên dịch viên phụ đề phim chuyên nghiệp, dịch từ {s} sang {t}.\n"
        "Bạn nhận một JSON gồm: context_note (ghi chú của người dùng về nội dung), "
        "previous_lines (các câu ngay trước đó, kèm bản dịch đã có), next_lines (các câu ngay sau đó) "
        "và translate_these (danh sách {id, text} cần dịch).\n"
        "Quy tắc:\n"
        "- Dịch theo NGỮ CẢNH của cả đoạn hội thoại: giữ nhất quán tên riêng, cách xưng hô, "
        "đại từ, giọng điệu giữa các câu; hiểu câu bị cắt dở bằng các câu lân cận.\n"
        "- Dịch tự nhiên như người bản ngữ, ngắn gọn, độ dài tương đương câu gốc vì bản dịch sẽ được đọc thành lời.\n"
        "- Không giải thích, không thêm ghi chú, không gộp hay tách câu.\n"
        '- Chỉ trả về JSON dạng {"translations":[{"id":<số>,"text":"<bản dịch>"}]} '
        "với đúng các id trong translate_these. Không dịch previous_lines và next_lines."
    )
    return sys


def _llm(segs, src, tgt, note, call, progress, cancel, log, batch=30):
    system = _system_prompt(src, tgt, note)
    out = [None] * len(segs)
    for start in range(0, len(segs), batch):
        if cancel.is_set():
            raise Cancelled()
        end = min(start + batch, len(segs))
        items = [{"id": k, "text": segs[k].text} for k in range(start, end)]
        before = [{"original": segs[j].text, "translated": out[j]}
                  for j in range(max(0, start - 4), start) if out[j]]
        after = [segs[j].text for j in range(end, min(len(segs), end + 3))]

        got = {}
        todo = items
        for _ in range(3):
            prompt = json.dumps({"context_note": note or "", "previous_lines": before,
                                 "next_lines": after, "translate_these": todo}, ensure_ascii=False)
            raw = call(system, prompt, cancel, log)
            raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", (raw or "").strip())
            try:
                data = json.loads(raw)
                lst = data.get("translations", []) if isinstance(data, dict) else data
                for it in lst:
                    txt = str(it.get("text", "")).strip()
                    if txt:
                        got[int(it["id"])] = txt
            except Exception:
                pass
            todo = [it for it in items if it["id"] not in got]
            if not todo:
                break
        for k in range(start, end):
            if k in got:
                out[k] = got[k]
            else:
                out[k] = segs[k].text
                log(f"✖ Không dịch được câu {k + 1}, giữ nguyên câu gốc.")
        progress(int(end / len(segs) * 100))
    return _copy_with(segs, out)


def _gemini_caller(api_key, model):
    model = (model or "gemini-3.8-flash").strip()
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    safety = [{"category": c, "threshold": "BLOCK_NONE"} for c in (
        "HARM_CATEGORY_HARASSMENT", "HARM_CATEGORY_HATE_SPEECH",
        "HARM_CATEGORY_SEXUALLY_EXPLICIT", "HARM_CATEGORY_DANGEROUS_CONTENT")]

    def call(system, prompt, cancel, log):
        body = json.dumps({
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {"responseMimeType": "application/json", "temperature": 0.3},
            "safetySettings": safety,
        }).encode("utf-8")
        for attempt in range(6):
            req = urllib.request.Request(url, data=body, headers={
                "Content-Type": "application/json", "x-goog-api-key": api_key})
            try:
                with urllib.request.urlopen(req, timeout=120) as r:
                    d = json.loads(r.read().decode("utf-8"))
                cands = d.get("candidates") or []
                if not cands:
                    log(f"Gemini không trả kết quả cho cụm này ({d.get('promptFeedback', '')}).")
                    return ""
                parts = (cands[0].get("content") or {}).get("parts") or []
                return "".join(p.get("text", "") for p in parts)
            except urllib.error.HTTPError as e:
                msg = e.read().decode("utf-8", "replace")[:300]
                if e.code in (429, 500, 503) and attempt < 5:
                    wait = min(2 ** attempt * 5, 60)
                    log(f"Gemini báo bận/giới hạn tốc độ (mã {e.code}), chờ {wait}s rồi thử lại...")
                    _sleep(wait, cancel)
                    continue
                if e.code == 429:
                    raise RuntimeError(
                        "Gemini hết hạn mức (429). Chờ một lúc, hoặc đổi sang model khác ở ô 'Model', "
                        f"hoặc kiểm tra hạn mức trong Google AI Studio. {msg}")
                if e.code in (400, 401, 403, 404):
                    raise RuntimeError(
                        f"Gemini từ chối (mã {e.code}). Kiểm tra API key và tên model '{model}'. {msg}")
                raise RuntimeError(f"Gemini lỗi {e.code}: {msg}")
            except urllib.error.URLError as e:
                if attempt < 5:
                    _sleep(3, cancel)
                    continue
                raise RuntimeError(f"Không kết nối được tới Gemini: {e.reason}")
        return ""

    return call


OLLAMA_URL = "http://127.0.0.1:11434"


def _ollama_http(path, payload=None, timeout=900):
    # bỏ qua proxy hệ thống: Ollama chạy ngay trên máy này
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(OLLAMA_URL + path, data=data,
                                 headers={"Content-Type": "application/json"})
    with opener.open(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def _ollama_caller(model):
    model = (model or "qwen2.5:7b").strip()
    try:
        tags = _ollama_http("/api/tags", timeout=5)
    except Exception:
        raise RuntimeError(
            f"Không kết nối được Ollama ({OLLAMA_URL}). Hãy cài Ollama từ ollama.com, "
            "mở ứng dụng Ollama (biểu tượng ở khay hệ thống) rồi thử lại.")
    names = {m.get("name", "") for m in tags.get("models", [])}
    if model not in names and f"{model}:latest" not in names:
        raise RuntimeError(
            f"Ollama chưa có model '{model}'. Mở Command Prompt và chạy: ollama pull {model}")

    def call(system, prompt, cancel, log):
        if cancel.is_set():
            raise Cancelled()
        try:
            d = _ollama_http("/api/chat", {
                "model": model, "stream": False, "format": "json",
                "options": {"temperature": 0.3, "num_ctx": 8192},
                "messages": [{"role": "system", "content": system},
                             {"role": "user", "content": prompt}],
            })
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"Ollama lỗi {e.code}: {e.read().decode('utf-8', 'replace')[:200]}")
        except Exception as e:
            raise RuntimeError(f"Ollama không phản hồi: {e}")
        return (d.get("message") or {}).get("content", "")

    return call


GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"


def _groq_caller(keys, model):
    """Gọi Groq với nhiều API key: mỗi cụm dùng key kế tiếp; key bị 429 thì nhảy sang key khác."""
    keys = list(dict.fromkeys(k.strip() for k in (keys or []) if k and k.strip()))
    if not keys:
        raise RuntimeError("Chưa nhập API key Groq (mỗi dòng một key).")
    model = (model or "llama-3.3-70b-versatile").strip()
    state = {"i": 0, "json": True}
    tag = lambda k: "…" + k[-4:]

    def once(key, system, prompt):
        body = {"model": model, "temperature": 0.3,
                "messages": [{"role": "system", "content": system},
                             {"role": "user", "content": prompt}]}
        if state["json"]:
            body["response_format"] = {"type": "json_object"}
        req = urllib.request.Request(GROQ_URL, data=json.dumps(body).encode("utf-8"), headers={
            "Content-Type": "application/json", "Authorization": f"Bearer {key}",
            "User-Agent": "VideoAITool/1.0"})
        with urllib.request.urlopen(req, timeout=120) as r:
            d = json.loads(r.read().decode("utf-8"))
        return ((d.get("choices") or [{}])[0].get("message") or {}).get("content", "") or ""

    def call(system, prompt, cancel, log):
        rounds = 0
        while True:
            waits = []
            for _ in range(len(keys)):
                if cancel.is_set():
                    raise Cancelled()
                idx = state["i"] % len(keys)
                key = keys[idx]
                try:
                    out = once(key, system, prompt)
                    state["i"] = idx + 1                     # lần sau dùng key kế tiếp
                    return out
                except urllib.error.HTTPError as e:
                    msg = e.read().decode("utf-8", "replace")[:300]
                    if e.code == 429:
                        try:
                            waits.append(float(e.headers.get("retry-after") or 0))
                        except ValueError:
                            waits.append(0.0)
                        log(f"Key {tag(key)} hết hạn mức, chuyển sang key khác.")
                        state["i"] = idx + 1
                    elif e.code in (401, 403):
                        log(f"Key {tag(key)} bị từ chối (mã {e.code}), loại khỏi danh sách.")
                        keys.pop(idx)
                        state["i"] = idx
                        if not keys:
                            raise RuntimeError(f"Tất cả API key Groq đều không dùng được. {msg}")
                    elif e.code == 400 and state["json"] and "response_format" in msg:
                        state["json"] = False
                        log("Model này không hỗ trợ chế độ JSON, thử lại ở chế độ thường.")
                    elif e.code >= 500:
                        waits.append(3.0)
                        log(f"Groq lỗi máy chủ (mã {e.code}), thử key khác.")
                        state["i"] = idx + 1
                    else:
                        raise RuntimeError(
                            f"Groq lỗi {e.code} (kiểm tra tên model '{model}'). {msg}")
                except urllib.error.URLError as e:
                    waits.append(3.0)
                    log(f"Không kết nối được Groq ({e.reason}), thử lại.")
                    state["i"] = idx + 1
            rounds += 1
            if rounds > 6:
                raise RuntimeError(
                    "Groq không phản hồi sau nhiều lần thử (các key có thể đã hết hạn mức). "
                    "Thêm key khác, đổi model hoặc thử lại sau.")
            wait = min(max(waits) if waits and max(waits) > 0 else 8 * rounds, 65)
            log(f"Tất cả key đang bị giới hạn, chờ {int(wait)}s...")
            _sleep(wait, cancel)

    log_msg = f"Groq: {len(keys)} API key xoay vòng, model {model}."
    call.info = log_msg
    return call


def _openai_caller(api_key, model):
    from openai import OpenAI

    client = OpenAI(api_key=api_key)

    def call(system, prompt, cancel, log):
        r = client.chat.completions.create(
            model=model, response_format={"type": "json_object"}, temperature=0.3,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}])
        return r.choices[0].message.content

    return call


def translate_segments(segs, src, tgt, o, progress, cancel, log):
    """o: pipeline.Options (dùng tr_engine, gemini_*, openai_*, context_note)."""
    if not segs or src == tgt:
        return segs
    if o.tr_engine == "groq":
        call = _groq_caller(o.groq_keys, o.groq_model)
        log(call.info)
        # cụm nhỏ hơn để mỗi yêu cầu tốn ít token, ít chạm giới hạn tokens/phút của từng key
        return _llm(segs, src, tgt, o.context_note, call, progress, cancel, log, batch=20)
    if o.tr_engine == "gemini":
        call = _gemini_caller(o.gemini_key, o.gemini_model)
        return _llm(segs, src, tgt, o.context_note, call, progress, cancel, log)
    if o.tr_engine == "ollama":
        call = _ollama_caller(o.ollama_model)
        # model chạy trên máy thường nhỏ hơn nên chia cụm nhỏ hơn để dịch chính xác hơn
        return _llm(segs, src, tgt, o.context_note, call, progress, cancel, log, batch=10)
    if o.tr_engine == "openai":
        call = _openai_caller(o.openai_key, o.openai_model)
        return _llm(segs, src, tgt, o.context_note, call, progress, cancel, log)
    return _google(segs, src, tgt, progress, cancel, log)
