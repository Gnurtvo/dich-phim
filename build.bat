@echo off
REM Đóng gói thành VideoAITool.exe. Cần đặt ffmpeg.exe và ffprobe.exe vào thư mục "ffmpeg" trước.
pyinstaller --noconfirm --windowed --name VideoAITool ^
  --collect-all faster_whisper --collect-all ctranslate2 ^

  --add-binary "ffmpeg\ffmpeg.exe;ffmpeg" --add-binary "ffmpeg\ffprobe.exe;ffmpeg" ^
  main.py
echo.
echo Xong. File chay: dist\VideoAITool\VideoAITool.exe
pause
