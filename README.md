# Video AI Tool

Phần mềm Windows (Python + PySide6): nhập video → nhận dạng giọng nói → dịch bằng AI → lồng tiếng → xóa metadata.

## Cài đặt (chạy từ mã nguồn)

1. Cài Python 3.10+ (tick "Add to PATH").
2. Tải FFmpeg (bản "essentials" từ gyan.dev), giải nén, rồi **một trong hai**:
   - thêm thư mục `bin` vào PATH, hoặc
   - copy `ffmpeg.exe`, `ffprobe.exe` vào thư mục `ffmpeg\` cạnh `main.py`.
3. Cài thư viện: `pip install -r requirements.txt`
4. Chạy: `python main.py`

Đóng gói thành .exe: chạy `build.bat` (kết quả ở `dist\VideoAITool\`).

## Cấu trúc

| File | Vai trò |
|---|---|
| `ui.py` | Giao diện, kéo thả, hàng đợi nhiều video, nút Hủy |
| `pipeline.py` | Điều phối các bước cho từng video |
| `transcribe.py` | faster-whisper (offline) / OpenAI Whisper API |
| `translate.py` | Google (free), Argos (offline), OpenAI |
| `tts.py` | Edge TTS, Windows SAPI (offline), OpenAI TTS; tự tăng tốc câu quá dài để khớp thời gian |
| `ffmpeg_utils.py` | Trích âm thanh, trộn tiếng lồng, nhúng phụ đề, xóa metadata |

## Ghi chú

- **Offline hoàn toàn**: Whisper + Argos + SAPI. Lần đầu chạy, Whisper và gói Argos cần mạng để tải model. SAPI cần cài giọng tiếng Việt trong Windows (Settings → Time & Language → Speech).
- **GPU NVIDIA**: chọn thiết bị `cuda` (cần CUDA/cuDNN phù hợp với ctranslate2) để Whisper nhanh hơn nhiều.
- **Xóa metadata**: xóa tag container, chapters, handler name và dấu hiệu encoder. Video được copy luồng hình (không encode lại) nên nhanh và không giảm chất lượng. Metadata nhúng trong chính luồng video (ví dụ SEI của encoder) sẽ vẫn còn.
- **API key OpenAI** được lưu trong registry của Windows (QSettings), không mã hóa. Dùng máy cá nhân.
- Chỉ dịch/lồng tiếng video bạn có quyền sử dụng.

## Hướng phát triển tiếp

Tách nhiều giọng người nói (diarization), giữ nhạc nền bằng tách vocal (Demucs), clone giọng, burn phụ đề cứng, trình sửa phụ đề trước khi lồng tiếng.
