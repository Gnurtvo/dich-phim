import os
import sys

# Bản đóng gói --windowed không có console: sys.stdout/sys.stderr là None, nhiều thư viện
# (VieNeu, tqdm, huggingface_hub...) gọi print()/write() sẽ lỗi "'NoneType' object has no attribute 'write'".
if sys.stdout is None:
    sys.stdout = open(os.devnull, "w", encoding="utf-8")
if sys.stderr is None:
    sys.stderr = open(os.devnull, "w", encoding="utf-8")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TQDM_DISABLE", "1")

from ui import run

if __name__ == "__main__":
    run()
