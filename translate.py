"""Dịch các đoạn phụ đề bằng LLM có ngữ cảnh: Groq (nhiều API key xoay vòng) hoặc OpenAI."""
import json
import re
import time
import urllib.error
import urllib.request

from common import LANGS, Cancelled, Segment

LANG_NAMES = {code: name for name, code in LANGS}
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODELS_URL = "https://api.groq.com/openai/v1/models"
DEFAULT_GROQ_MODEL = "llama-3.3-70b-versatile"
# thứ tự ưu tiên khi model đang chọn không dùng được với tài khoản
PREFERRED_GROQ = ["llama-3.3-70b-versatile", "openai/gpt-oss-120b", "openai/gpt-oss-20b",
                  "llama-3.1-8b-instant", "qwen/qwen3.8-27b", "minimaxai/minimax-m2.7"]
_NON_CHAT = ("whisper", "tts", "orpheus", "guard", "safeguard", "embed", "playai")


def _copy_with(segs, texts):
    return [Segment(s.start, s.end, t) for s, t in zip(segs, texts)]


def _sleep(seconds, cancel):
    end = time.time() + seconds
    while time.time() < end:
        if cancel.is_set():
            raise Cancelled()
        time.sleep(0.1)


def _clean_keys(keys):
    return list(dict.fromkeys(k.strip() for k in (keys or []) if k and k.strip()))


def _tag(key):
    return "…" + key[-4:]


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
            raw = re.sub(r"<think>.*?</think>", "", raw or "", flags=re.S).strip()   # model suy luận
            raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw)
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


# ---------------- Groq: nhiều key xoay vòng ----------------
def _groq_headers(key):
    # Groq nằm sau Cloudflare, cần User-Agent riêng (mặc định của Python bị chặn)
    return {"Content-Type": "application/json", "Authorization": f"Bearer {key}",
            "User-Agent": "VideoAITool/1.0"}


def list_groq_models(keys):
    """Hỏi Groq tài khoản này dùng được những model chat nào (thử lần lượt từng key)."""
    for key in _clean_keys(keys):
        req = urllib.request.Request(GROQ_MODELS_URL, headers=_groq_headers(key))
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                data = json.loads(r.read().decode("utf-8")).get("data", [])
        except Exception:
            continue
        ids = [m["id"] for m in data
               if m.get("id") and m.get("active", True) and not any(x in m["id"].lower() for x in _NON_CHAT)]
        if ids:
            return sorted(ids)
    return []


def pick_groq_model(available, exclude=()):
    """Chọn model tốt nhất trong danh sách có sẵn (theo PREFERRED_GROQ), bỏ qua các model đã thử."""
    cands = [m for m in available if m not in exclude]
    for m in PREFERRED_GROQ:
        if m in cands:
            return m
    return cands[0] if cands else None


def _groq_caller(keys, model):
    """Mỗi cụm dùng key kế tiếp; key bị 429 thì nhảy sang key khác; key sai thì loại bỏ."""
    keys = _clean_keys(keys)
    if not keys:
        raise RuntimeError("Chưa nhập API key Groq (mỗi dòng một key).")
    model = (model or DEFAULT_GROQ_MODEL).strip()
    state = {"i": 0, "json": True, "model": model, "tried": {model}}

    def once(key, system, prompt):
        body = {"model": state["model"], "temperature": 0.3,
                "messages": [{"role": "system", "content": system},
                             {"role": "user", "content": prompt}]}
        if state["json"]:
            body["response_format"] = {"type": "json_object"}
        req = urllib.request.Request(GROQ_URL, data=json.dumps(body).encode("utf-8"),
                                     headers=_groq_headers(key))
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
                        log(f"Key {_tag(key)} hết hạn mức, chuyển sang key khác.")
                        state["i"] = idx + 1
                    elif e.code in (401, 403):
                        log(f"Key {_tag(key)} bị từ chối (mã {e.code}), loại khỏi danh sách.")
                        keys.pop(idx)
                        state["i"] = idx
                        if not keys:
                            raise RuntimeError(f"Tất cả API key Groq đều không dùng được. {msg}")
                    elif e.code in (400, 404) and "model" in msg.lower() and any(
                            w in msg.lower() for w in ("does not exist", "not_found", "decommission")):
                        # tài khoản không có quyền với model này: hỏi Groq xem dùng được model nào và chuyển sang
                        avail = list_groq_models([key])
                        new = pick_groq_model(avail, exclude=state["tried"])
                        if not new:
                            raise RuntimeError(
                                f"Tài khoản Groq không dùng được model '{state['model']}' và không có model "
                                "thay thế. Model có sẵn: " + (", ".join(avail) or "(không lấy được danh sách)"))
                        log(f"Model '{state['model']}' không có trong tài khoản Groq, tự chuyển sang '{new}'.")
                        state["model"] = new
                        state["tried"].add(new)
                        state["json"] = True
                    elif e.code == 400 and state["json"] and "response_format" in msg:
                        state["json"] = False
                        log("Model này không hỗ trợ chế độ JSON, thử lại ở chế độ thường.")
                    elif e.code >= 500:
                        waits.append(3.0)
                        log(f"Groq lỗi máy chủ (mã {e.code}), thử key khác.")
                        state["i"] = idx + 1
                    else:
                        raise RuntimeError(
                            f"Groq lỗi {e.code} (model '{state['model']}'). {msg}")
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

    call.info = f"Groq: {len(keys)} API key xoay vòng, model {model}."
    return call


def check_groq_keys(keys, model=""):
    """Gửi một yêu cầu nhỏ bằng từng key. Trả về [(số thứ tự, '…abcd', mức, mô tả)],
    mức là 'ok' (dùng được), 'warn' (key đúng nhưng đang hết hạn mức) hoặc 'fail'."""
    model = (model or DEFAULT_GROQ_MODEL).strip()
    results = []
    for n, key in enumerate(_clean_keys(keys), 1):
        body = {"model": model, "temperature": 0, "max_tokens": 30,
                "messages": [{"role": "user",
                              "content": "Dịch sang tiếng Việt, chỉ trả lời bản dịch: Hello, how are you?"}]}
        req = urllib.request.Request(GROQ_URL, data=json.dumps(body).encode("utf-8"),
                                     headers=_groq_headers(key))
        t0 = time.time()
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                d = json.loads(r.read().decode("utf-8"))
            text = (((d.get("choices") or [{}])[0].get("message") or {}).get("content") or "").strip()
            results.append((n, _tag(key), "ok",
                            f"dùng được ({time.time() - t0:.1f}s), model trả lời: “{text[:50]}”"))
        except urllib.error.HTTPError as e:
            raw = e.read().decode("utf-8", "replace")
            try:
                detail = json.loads(raw)["error"]["message"]
            except Exception:
                detail = raw[:150]
            if e.code == 429:
                ra = e.headers.get("retry-after")
                results.append((n, _tag(key), "warn",
                                "key đúng nhưng đang hết hạn mức" + (f" (thử lại sau ~{ra}s)" if ra else "")))
            elif e.code in (401, 403):
                results.append((n, _tag(key), "fail", f"key sai hoặc bị thu hồi/khóa (mã {e.code}): {detail}"))
            elif e.code == 404 or (e.code == 400 and "model" in detail.lower()):
                results.append((n, _tag(key), "warn",
                                f"key hợp lệ nhưng model '{model}' không có hoặc tài khoản chưa được cấp quyền "
                                f"(chọn model khác trong danh sách): {detail}"))
            else:
                results.append((n, _tag(key), "fail", f"lỗi {e.code}: {detail}"))
        except urllib.error.URLError as e:
            results.append((n, _tag(key), "fail", f"không kết nối được Groq: {e.reason}"))
        except Exception as e:
            results.append((n, _tag(key), "fail", f"lỗi: {e}"))
    return results


# ---------------- OpenAI ----------------
def _openai_caller(api_key, model):
    from openai import OpenAI

    client = OpenAI(api_key=api_key)

    def call(system, prompt, cancel, log):
        r = client.chat.completions.create(
            model=model, response_format={"type": "json_object"}, temperature=0.3,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}])
        return r.choices[0].message.content

    return call


# ---------------- Bước phân tích trước: cốt truyện, nhân vật, xưng hô ----------------
BRIEF_MAX_CHARS = 4500      # giới hạn lượng lời thoại gửi đi phân tích (tiết kiệm token/phút của Groq)
BRIEF_NOTE_CHARS = 1800     # giới hạn độ dài tài liệu tham khảo gửi kèm mỗi cụm dịch


def _loads_lenient(raw):
    raw = re.sub(r"<think>.*?</think>", "", raw or "", flags=re.S).strip()
    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw)
    for cand in (raw, raw[raw.find("{"):raw.rfind("}") + 1]):
        try:
            return json.loads(cand)
        except Exception:
            continue
    return None


def _sample_script(segs):
    """Lời thoại đánh số; nếu quá dài thì lấy mẫu đều trên toàn bộ video."""
    lines = [f"{i + 1}. {s.text}" for i, s in enumerate(segs)]
    total = sum(len(x) + 1 for x in lines)
    if total > BRIEF_MAX_CHARS:
        keep = max(8, int(len(lines) * BRIEF_MAX_CHARS / total))
        step = len(lines) / keep
        lines = [lines[int(k * step)] for k in range(keep)]
    return "\n".join(lines)


def _format_brief(d):
    if not isinstance(d, dict):
        return ""
    out = []
    if d.get("summary"):
        out.append(f"Tóm tắt: {d['summary']}")
    chars = []
    for c in d.get("characters") or []:
        if isinstance(c, dict) and c.get("name"):
            extra = ", ".join(str(x) for x in (c.get("gender"), c.get("role")) if x and x != "không rõ")
            chars.append(f"{c['name']} → {c.get('target') or c['name']}" + (f" ({extra})" if extra else ""))
    if chars:
        out.append("Nhân vật: " + "; ".join(chars))
    if d.get("address_terms"):
        out.append(f"Xưng hô: {d['address_terms']}")
    gl = [f"{g['src']} → {g['target']}" for g in (d.get("glossary") or [])
          if isinstance(g, dict) and g.get("src") and g.get("target")]
    if gl:
        out.append("Thuật ngữ/tên riêng: " + "; ".join(gl))
    if d.get("style"):
        out.append(f"Giọng điệu: {d['style']}")
    return "\n".join(out)[:BRIEF_NOTE_CHARS]


def make_brief(segs, src, tgt, user_note, call, cancel, log):
    """Đọc toàn bộ lời thoại một lượt để rút ra tóm tắt, nhân vật, xưng hô, thuật ngữ. Lỗi thì bỏ qua."""
    s, t = LANG_NAMES.get(src, src), LANG_NAMES.get(tgt, tgt)
    system = (
        f"Bạn là biên tập viên phụ đề. Bạn nhận toàn bộ lời thoại {s} của một video (các câu có đánh số, "
        f"có thể có lỗi nhận dạng giọng nói) để lập TÀI LIỆU THAM KHẢO phục vụ việc dịch sang {t}.\n"
        "Chỉ trả về một JSON dạng: {\"summary\": \"tóm tắt 2-4 câu: thể loại, bối cảnh, diễn biến\", "
        "\"characters\": [{\"name\": \"tên/cách gọi trong bản gốc\", \"target\": \"cách gọi nên dùng trong bản dịch\", "
        "\"gender\": \"nam/nữ/không rõ\", \"role\": \"vai trò hoặc quan hệ\"}], "
        "\"address_terms\": \"quy ước xưng hô giữa các nhân vật trong bản dịch (vd: vợ-chồng gọi nhau anh/em)\", "
        "\"glossary\": [{\"src\": \"...\", \"target\": \"...\"}], \"style\": \"giọng điệu chung\"}.\n"
        "Chỉ ghi điều có cơ sở trong lời thoại; không chắc thì ghi \"không rõ\" hoặc bỏ trống, không bịa."
    )
    prompt = json.dumps({"ghi_chu_nguoi_dung": user_note or "", "loi_thoai": _sample_script(segs)},
                        ensure_ascii=False)
    try:
        data = _loads_lenient(call(system, prompt, cancel, log))
        return _format_brief(data)
    except Cancelled:
        raise
    except Exception as e:
        log(f"Không phân tích trước được cốt truyện ({str(e)[:80]}), dịch bình thường.")
        return ""


def translate_segments(segs, src, tgt, o, progress, cancel, log):
    """o: pipeline.Options (dùng tr_engine, groq_*, openai_*, context_note, use_brief)."""
    if not segs or src == tgt:
        return segs
    if o.tr_engine == "openai":
        call = _openai_caller(o.openai_key, o.openai_model)
        batch = 30
    else:
        call = _groq_caller(o.groq_keys, o.groq_model)
        log(call.info)
        batch = 20      # cụm nhỏ để mỗi yêu cầu tốn ít token, ít chạm giới hạn tokens/phút của từng key
    note = o.context_note
    if getattr(o, "use_brief", True) and len(segs) >= 8:
        log("Phân tích trước cốt truyện, nhân vật và cách xưng hô...")
        brief = make_brief(segs, src, tgt, o.context_note, call, cancel, log)
        if brief:
            log("Tài liệu tham khảo AI tự lập (dùng cho cả quá trình dịch):\n" + brief)
            note = (note + "\n" if note else "") + "[Tài liệu tham khảo tự lập từ toàn bộ lời thoại]\n" + brief
    return _llm(segs, src, tgt, note, call, progress, cancel, log, batch=batch)
