"""Giao diện PySide6."""
import threading
from pathlib import Path

from PySide6.QtCore import QSettings, Qt, QThread, Signal
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout, QGroupBox,
    QHBoxLayout, QLabel, QLineEdit, QListWidget, QMessageBox, QPlainTextEdit, QProgressBar,
    QPushButton, QSlider, QVBoxLayout, QWidget,
)

from common import EDGE_VOICES, LANGS, OPENAI_VOICES
from pipeline import Options, process_video

VIDEO_EXT = (".mp4", ".mkv", ".mov", ".avi", ".webm", ".flv", ".wmv", ".m4v", ".ts")


class Worker(QThread):
    log = Signal(str)
    progress = Signal(int)
    done = Signal(bool, str)

    def __init__(self, files, opts):
        super().__init__()
        self.files, self.opts = files, opts
        self.cancel = threading.Event()

    def run(self):
        n = len(self.files)
        errors = 0
        try:
            for i, f in enumerate(self.files):
                self.log.emit(f"\n=== [{i + 1}/{n}] {Path(f).name} ===")
                try:
                    process_video(
                        f, self.opts, self.log.emit,
                        lambda p, i=i: self.progress.emit(int((i + p / 100) / n * 100)),
                        self.cancel)
                except Exception as e:
                    from common import Cancelled
                    if isinstance(e, Cancelled):
                        raise
                    errors += 1
                    self.log.emit(f"✖ Lỗi: {e}")
            self.done.emit(True, f"Xong. {n - errors}/{n} video thành công.")
        except Exception:
            self.done.emit(False, "Đã hủy.")


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Video AI Tool – Dịch, lồng tiếng, xóa metadata")
        self.resize(980, 720)
        self.setAcceptDrops(True)
        self.settings = QSettings("VideoAITool", "App")
        self.worker = None
        self._build()
        self._load()

    # ---------- dựng giao diện ----------
    def _build(self):
        root = QHBoxLayout(self)

        # cột trái: danh sách file + log
        left = QVBoxLayout()
        self.files = QListWidget()
        self.files.setSelectionMode(QListWidget.ExtendedSelection)
        btns = QHBoxLayout()
        b_add = QPushButton("Thêm video...")
        b_add.clicked.connect(self.add_files)
        b_del = QPushButton("Xóa mục chọn")
        b_del.clicked.connect(lambda: [self.files.takeItem(self.files.row(i)) for i in self.files.selectedItems()])
        b_clr = QPushButton("Xóa hết")
        b_clr.clicked.connect(self.files.clear)
        for b in (b_add, b_del, b_clr):
            btns.addWidget(b)
        left.addWidget(QLabel("Video (có thể kéo thả vào đây):"))
        left.addWidget(self.files, 2)
        left.addLayout(btns)

        out = QHBoxLayout()
        self.out_dir = QLineEdit()
        self.out_dir.setPlaceholderText("Mặc định: cùng thư mục với video gốc")
        b_out = QPushButton("Chọn thư mục xuất...")
        b_out.clicked.connect(self.pick_out)
        out.addWidget(self.out_dir)
        out.addWidget(b_out)
        left.addLayout(out)

        self.bar = QProgressBar()
        self.logbox = QPlainTextEdit()
        self.logbox.setReadOnly(True)
        left.addWidget(self.bar)
        left.addWidget(self.logbox, 3)
        run = QHBoxLayout()
        self.b_start = QPushButton("▶ Bắt đầu")
        self.b_start.clicked.connect(self.start)
        self.b_stop = QPushButton("■ Hủy")
        self.b_stop.setEnabled(False)
        self.b_stop.clicked.connect(self.stop)
        run.addWidget(self.b_start)
        run.addWidget(self.b_stop)
        left.addLayout(run)
        root.addLayout(left, 3)

        # cột phải: tùy chọn
        right = QVBoxLayout()

        g = QGroupBox("1. Nhận dạng giọng nói")
        f = QFormLayout(g)
        self.c_asr = QCheckBox("Xuất phụ đề gốc (.srt)")
        self.c_asr.setChecked(True)
        self.asr_engine = QComboBox()
        self.asr_engine.addItem("Whisper offline (trên máy)", "local")
        self.asr_engine.addItem("OpenAI Whisper API", "openai")
        self.model = QComboBox()
        self.model.addItems(["tiny", "base", "small", "medium", "large-v3"])
        self.model.setCurrentText("small")
        self.device = QComboBox()
        self.device.addItems(["auto", "cpu", "cuda"])
        self.src = QComboBox()
        self.src.addItem("Tự nhận diện", "auto")
        for n, c in LANGS:
            self.src.addItem(n, c)
        f.addRow(self.c_asr)
        f.addRow("Engine:", self.asr_engine)
        f.addRow("Model:", self.model)
        f.addRow("Thiết bị:", self.device)
        f.addRow("Ngôn ngữ gốc:", self.src)
        right.addWidget(g)

        g = QGroupBox("2. Dịch bằng AI")
        f = QFormLayout(g)
        self.c_tr = QCheckBox("Dịch phụ đề")
        self.c_tr.setChecked(True)
        self.tr_engine = QComboBox()
        self.tr_engine.addItem("Google (miễn phí, cần mạng)", "google")
        self.tr_engine.addItem("Argos (offline)", "argos")
        self.tr_engine.addItem("OpenAI (chất lượng cao)", "openai")
        self.tgt = QComboBox()
        for n, c in LANGS:
            self.tgt.addItem(n, c)
        self.oa_model = QLineEdit("gpt-4o-mini")
        f.addRow(self.c_tr)
        f.addRow("Engine:", self.tr_engine)
        f.addRow("Dịch sang:", self.tgt)
        f.addRow("Model OpenAI:", self.oa_model)
        right.addWidget(g)

        g = QGroupBox("3. Lồng tiếng")
        f = QFormLayout(g)
        self.c_dub = QCheckBox("Lồng tiếng (TTS)")
        self.c_dub.setChecked(True)
        self.tts_engine = QComboBox()
        self.tts_engine.addItem("Edge TTS (giọng tự nhiên, cần mạng)", "edge")
        self.tts_engine.addItem("Windows SAPI (offline)", "sapi")
        self.tts_engine.addItem("OpenAI TTS", "openai")
        self.voice = QComboBox()
        self.voice.setEditable(True)
        self.speed = QDoubleSpinBox()
        self.speed.setRange(0.5, 2.0)
        self.speed.setSingleStep(0.1)
        self.speed.setValue(1.0)
        self.vol = QSlider(Qt.Horizontal)
        self.vol.setRange(0, 100)
        self.vol.setValue(10)
        self.c_embed = QCheckBox("Nhúng phụ đề mềm vào video")
        f.addRow(self.c_dub)
        f.addRow("Engine:", self.tts_engine)
        f.addRow("Giọng:", self.voice)
        f.addRow("Tốc độ đọc:", self.speed)
        f.addRow("Âm gốc còn lại (%):", self.vol)
        f.addRow(self.c_embed)
        right.addWidget(g)

        g = QGroupBox("4. Quyền riêng tư")
        f = QFormLayout(g)
        self.c_strip = QCheckBox("Xóa metadata video (thông tin thiết bị, vị trí, thời gian...)")
        self.c_strip.setChecked(True)
        f.addRow(self.c_strip)
        right.addWidget(g)

        g = QGroupBox("OpenAI API key (nếu dùng)")
        f = QFormLayout(g)
        self.key = QLineEdit()
        self.key.setEchoMode(QLineEdit.Password)
        f.addRow(self.key)
        right.addWidget(g)
        right.addStretch(1)
        root.addLayout(right, 2)

        self.tts_engine.currentIndexChanged.connect(self._fill_voices)
        self.tgt.currentIndexChanged.connect(self._fill_voices)
        self._fill_voices()

    def _fill_voices(self):
        eng = self.tts_engine.currentData()
        cur = self.voice.currentText()
        self.voice.clear()
        if eng == "edge":
            self.voice.addItems(EDGE_VOICES.get(self.tgt.currentData(), []))
        elif eng == "openai":
            self.voice.addItems(OPENAI_VOICES)
        else:
            self.voice.setEditText("")
            self.voice.lineEdit().setPlaceholderText("Một phần tên giọng SAPI (vd: An, Hoai My) – để trống = mặc định")
        if cur and eng == "sapi":
            self.voice.setEditText(cur)

    # ---------- xử lý sự kiện ----------
    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e):
        self._add([u.toLocalFile() for u in e.mimeData().urls()])

    def _add(self, paths):
        have = {self.files.item(i).text() for i in range(self.files.count())}
        for p in paths:
            if p.lower().endswith(VIDEO_EXT) and p not in have:
                self.files.addItem(p)

    def add_files(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Chọn video", "", "Video (*" + " *".join(VIDEO_EXT) + ")")
        self._add(paths)

    def pick_out(self):
        d = QFileDialog.getExistingDirectory(self, "Thư mục xuất")
        if d:
            self.out_dir.setText(d)

    def _options(self):
        return Options(
            output_dir=self.out_dir.text().strip(),
            do_transcribe=self.c_asr.isChecked(),
            do_translate=self.c_tr.isChecked(),
            do_dub=self.c_dub.isChecked(),
            embed_subs=self.c_embed.isChecked(),
            strip_metadata=self.c_strip.isChecked(),
            asr_engine=self.asr_engine.currentData(),
            whisper_model=self.model.currentText(),
            device=self.device.currentText(),
            src_lang=self.src.currentData(),
            tr_engine=self.tr_engine.currentData(),
            tgt_lang=self.tgt.currentData(),
            openai_model=self.oa_model.text().strip() or "gpt-4o-mini",
            tts_engine=self.tts_engine.currentData(),
            voice=self.voice.currentText().strip(),
            tts_speed=self.speed.value(),
            orig_volume=self.vol.value() / 100,
            openai_key=self.key.text().strip(),
        )

    def start(self):
        files = [self.files.item(i).text() for i in range(self.files.count())]
        if not files:
            QMessageBox.information(self, "Thiếu video", "Hãy thêm ít nhất một video.")
            return
        o = self._options()
        if not any((o.do_transcribe, o.do_translate, o.do_dub, o.embed_subs, o.strip_metadata)):
            QMessageBox.information(self, "Chưa chọn tác vụ", "Hãy bật ít nhất một tính năng.")
            return
        self._save()
        self.logbox.clear()
        self.bar.setValue(0)
        self.worker = Worker(files, o)
        self.worker.log.connect(self.logbox.appendPlainText)
        self.worker.progress.connect(self.bar.setValue)
        self.worker.done.connect(self._finished)
        self.b_start.setEnabled(False)
        self.b_stop.setEnabled(True)
        self.worker.start()

    def stop(self):
        if self.worker:
            self.worker.cancel.set()
            self.logbox.appendPlainText("Đang hủy...")

    def _finished(self, ok, msg):
        self.logbox.appendPlainText("\n" + msg)
        if ok:
            self.bar.setValue(100)
        self.b_start.setEnabled(True)
        self.b_stop.setEnabled(False)

    # ---------- lưu/đọc cài đặt ----------
    def _save(self):
        s = self.settings
        s.setValue("key", self.key.text())
        s.setValue("out_dir", self.out_dir.text())
        s.setValue("model", self.model.currentText())
        s.setValue("tgt", self.tgt.currentIndex())
        s.setValue("tr_engine", self.tr_engine.currentIndex())
        s.setValue("tts_engine", self.tts_engine.currentIndex())

    def _load(self):
        s = self.settings
        self.key.setText(s.value("key", ""))
        self.out_dir.setText(s.value("out_dir", ""))
        self.model.setCurrentText(s.value("model", "small"))
        self.tgt.setCurrentIndex(int(s.value("tgt", 0)))
        self.tr_engine.setCurrentIndex(int(s.value("tr_engine", 0)))
        self.tts_engine.setCurrentIndex(int(s.value("tts_engine", 0)))
        self._fill_voices()

    def closeEvent(self, e):
        if self.worker and self.worker.isRunning():
            self.worker.cancel.set()
            self.worker.wait(3000)
        e.accept()


def run():
    app = QApplication([])
    w = MainWindow()
    w.show()
    app.exec()
