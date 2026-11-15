"""In ra các cờ --collect-all cho vieneu và mọi thư viện nó phụ thuộc (để PyInstaller gộp cả file dữ liệu)."""
import importlib.metadata as m
import re

SKIP = {"torch", "torchaudio", "torchvision", "transformers", "numpy", "pip", "setuptools", "wheel"}


def top_levels(dist):
    try:
        txt = m.distribution(dist).read_text("top_level.txt")
        if txt:
            return [t.strip() for t in txt.split() if t.strip() and not t.startswith("_")]
    except Exception:
        pass
    return [dist.replace("-", "_")]


seen, todo, mods = set(), ["vieneu"], []
while todo:
    d = todo.pop()
    key = re.sub(r"[-_.]+", "-", d).lower()
    if key in seen or key in SKIP:
        continue
    seen.add(key)
    try:
        reqs = m.requires(d) or []
    except Exception:
        continue
    mods += top_levels(d)
    for r in reqs:
        if "extra ==" in r.replace("'", "").replace('"', ""):
            continue
        todo.append(re.split(r"[ ;<>=!~\[(]", r.strip())[0])
print(" ".join(f"--collect-all {x}" for x in dict.fromkeys(mods)))
