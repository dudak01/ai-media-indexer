import sys
import os
import time
from pathlib import Path

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QTextEdit, QFileDialog, QLabel, QTableWidget,
    QTableWidgetItem, QHeaderView, QProgressBar, QMessageBox, QLineEdit
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QTextCursor, QFont

try:
    from core.scanner import DirectoryScanner
    from core.extractor import TechnicalMetadataExtractor
    from core.enrichment import EnrichmentService
    from core.ner_predictor import NERPredictor
    from core.vector_db import VectorDatabase
    CORE_AVAILABLE = True
except ImportError as e:
    print(f"Критическая ошибка импорта: {e}")
    CORE_AVAILABLE = False


class MLWorker(QThread):
    log_signal = pyqtSignal(str)
    progress_signal = pyqtSignal(int)
    result_signal = pyqtSignal(list)
    finished_signal = pyqtSignal(object)

    def __init__(self, directory: str):
        super().__init__()
        self.directory = directory

    def run(self):
        self.log_signal.emit("[*] Загрузка мультимодального ИИ (CLIP, Shazam, FAISS)...")
        self.progress_signal.emit(5)

        try:
            scanner = DirectoryScanner(self.directory, compute_hashes=False)
            extractor = TechnicalMetadataExtractor()
            enricher = EnrichmentService()
            ner = NERPredictor()
            vector_db = VectorDatabase()

            inventory = scanner.scan()
            self.progress_signal.emit(20)

            all_files = []
            for m_type, m_list in inventory.items():
                all_files.extend([(f, m_type) for f in m_list])

            total = len(all_files)
            if total == 0:
                self.log_signal.emit("[!] Файлы не найдены.")
                self.finished_signal.emit(None)
                return

            gui_results = []
            for i, (media, ftype) in enumerate(all_files):
                self.log_signal.emit(f"[*] Анализ файла: {media.name}...")

                # 1. Технические метаданные
                extractor.extract(media.full_path, ftype)

                # 2. NER — извлекаем сущности из имени файла
                ner_result = ner.extract_entities(media.name)
                title = ner_result.get('title') or ''
                year = ner_result.get('year') or '---'
                quality = ner_result.get('quality') or '---'
                artist = ner_result.get('artist') or ''

                # 3. Обогащение через внешние сервисы
                enriched = enricher.enrich(media.full_path, ftype)

                # 4. Для видео — IMDb по NER-названию
                if ftype == 'video' and title:
                    imdb_data = enricher.enrich_video(title)
                    enriched.update(imdb_data)

                # 5. Итоговое отображаемое название
                stem = Path(media.name).stem
                if artist and title:
                    display_title = f"{artist} - {title}"
                elif title:
                    display_title = title
                else:
                    display_title = stem

                # 6. Формируем payload для FAISS
                payload = {
                    'real_file_name': media.name,
                    'display_title': display_title,
                    'year': year,
                    'quality': quality,
                    'ftype': ftype,
                }

                # 7. Добавляем в векторную базу
                if ftype == 'image':
                    vector_db.add_image(media.full_path, payload)
                    ocr_text = enriched.get('ocr_text', '')
                    if ocr_text:
                        vector_db.add_text(
                            f"изображение фото {ocr_text}",
                            payload
                        )
                elif ftype == 'audio':
                    parts = ['музыка аудио песня трек']
                    if artist:
                        parts.append(f"исполнитель {artist}")
                    if title:
                        parts.append(f"название {title}")
                    if quality and quality != '---':
                        parts.append(quality)
                    parts.append(stem)
                    search_text = " ".join(filter(bool, parts))
                    vector_db.add_text(search_text, payload)

                elif ftype == 'video':
                    parts = ['видео фильм кино']
                    if title:
                        parts.append(f"название {title}")
                    if year != '---':
                        parts.append(f"год {year}")
                    if quality != '---':
                        parts.append(quality)
                    if 'imdb' in enriched:
                        plot = enriched['imdb'].get('plot', '')
                        if plot:
                            parts.append(plot)
                    parts.append(stem)
                    search_text = " ".join(filter(bool, parts))
                    vector_db.add_text(search_text, payload)

                gui_results.append((media.name, display_title, year, quality, "Обогащено"))
                self.progress_signal.emit(20 + int((i + 1) / total * 80))
                self.log_signal.emit(f"[ИИ] Успешно: {media.name}")

            self.result_signal.emit(gui_results)
            self.finished_signal.emit(vector_db)

        except Exception as e:
            self.log_signal.emit(f"[ОШИБКА] {str(e)}")
            self.finished_signal.emit(None)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Интеллектуальный Индексатор Медиаконтента v2.0")
        self.setGeometry(100, 100, 1050, 800)
        self.vector_db = None
        self._init_ui()

    def _init_ui(self):
        main_widget = QWidget()
        main_layout = QVBoxLayout()

        top_panel = QHBoxLayout()
        self.btn_select_dir = QPushButton("📁 Выбрать папку")
        self.btn_start = QPushButton("🚀 Запустить Полную Индексацию")
        self.btn_select_dir.clicked.connect(self.select_directory)
        self.btn_start.clicked.connect(self.start_analysis)
        self.btn_start.setEnabled(False)
        top_panel.addWidget(self.btn_select_dir)
        top_panel.addWidget(self.btn_start)
        main_layout.addLayout(top_panel)

        self.progress_bar = QProgressBar()
        main_layout.addWidget(self.progress_bar)

        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(["Файл", "Смысловое Название", "Год", "Качество", "Статус"])
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        main_layout.addWidget(self.table)

        search_panel = QHBoxLayout()
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Введите запрос: 'грустная песня', 'фильм про роботов'...")
        self.search_input.setMinimumHeight(40)
        self.search_input.setFont(QFont("Arial", 11))

        self.btn_search = QPushButton("🔍 Семантический Поиск")
        self.btn_search.setMinimumHeight(40)
        self.btn_search.setStyleSheet("background-color: #ff9800; color: white; font-weight: bold;")
        self.btn_search.clicked.connect(self.perform_search)
        self.btn_search.setEnabled(False)

        search_panel.addWidget(self.search_input)
        search_panel.addWidget(self.btn_search)
        main_layout.addLayout(search_panel)

        self.console = QTextEdit()
        self.console.setReadOnly(True)
        self.console.setMaximumHeight(150)
        self.console.setStyleSheet("background-color: #1e1e1e; color: #00ff00; font-family: Consolas;")
        main_layout.addWidget(self.console)

        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)
        self.target_dir = ""

    def log_to_console(self, text: str):
        self.console.append(f"[{time.strftime('%H:%M:%S')}] {text}")
        self.console.moveCursor(QTextCursor.End)

    def select_directory(self):
        dir_path = QFileDialog.getExistingDirectory(self, "Выберите папку")
        if dir_path:
            self.target_dir = dir_path
            self.log_to_console(f"Выбрана директория: {dir_path}")
            self.btn_start.setEnabled(True)

    def start_analysis(self):
        self.btn_start.setEnabled(False)
        self.btn_search.setEnabled(False)
        self.table.setRowCount(0)
        self.progress_bar.setValue(0)

        self.worker = MLWorker(self.target_dir)
        self.worker.log_signal.connect(self.log_to_console)
        self.worker.progress_signal.connect(self.progress_bar.setValue)
        self.worker.result_signal.connect(self.populate_table)
        self.worker.finished_signal.connect(self.on_analysis_finished)
        self.worker.start()

    def populate_table(self, data: list):
        self.table.setRowCount(len(data))
        for r_idx, r_data in enumerate(data):
            for c_idx, text in enumerate(r_data):
                item = QTableWidgetItem(str(text))
                self.table.setItem(r_idx, c_idx, item)

    def on_analysis_finished(self, vector_db):
        self.btn_start.setEnabled(True)
        if vector_db:
            self.vector_db = vector_db
            self.btn_search.setEnabled(True)
            self.log_to_console("[*] База векторов создана. Можно выполнять семантический поиск!")

    def perform_search(self):
        query = self.search_input.text().strip()
        if not query or not self.vector_db:
            return

        self.log_to_console(f"\n[*] ПОИСК: '{query}' ...")

        raw_results = self.vector_db.search(query, k=5)

        # Разные пороги: фото требует более высокого сходства
        # чтобы аудио и видео не терялись на фоне CLIP-изображений
        valid_results = []
        for r in raw_results:
            if r['type'] == '📸 [ФОТО]' and r['score'] >= 60.0:
                valid_results.append(r)
            elif r['type'] != '📸 [ФОТО]' and r['score'] >= 45.0:
                valid_results.append(r)

        if not valid_results:
            self.log_to_console("[-] Совпадений с высокой уверенностью не найдено.")
            QMessageBox.warning(self, "Результат", "По вашему запросу ничего не найдено.\nПопробуйте переформулировать.")
            return

        res_text = "Найдены файлы (Уверенность ИИ > 45%):\n\n"
        for i, r in enumerate(valid_results[:3], 1):
            file_name = r['payload'].get('real_file_name', 'Неизвестный файл')
            res_text += f"{i}. {r['type']} {file_name} (Сходство: {r['score']}%)\n"
            self.log_to_console(f"[НАЙДЕНО] {r['type']} {file_name} ({r['score']}%)")

        QMessageBox.information(self, "Результаты поиска", res_text)


def run_gui():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    run_gui()