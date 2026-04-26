"""
Модуль графического пользовательского интерфейса (GUI) на базе PyQt5.
Обеспечивает взаимодействие пользователя с Интеллектуальным Ядром системы.
Реализует асинхронную обработку (паттерн Worker/QThread), чтобы
интерфейс не зависал во время тяжелых инференсов нейросети.

Тема ВКР: Индексация медиаконтента и обогащение метаданных с использованием интеллектуального анализа данных
"""

import sys
import time
from pathlib import Path
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QTextEdit, QFileDialog, QLabel, QTableWidget,
    QTableWidgetItem, QHeaderView, QProgressBar, QMessageBox
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont, QTextCursor

# Ленивый импорт ядра для ускорения запуска GUI
try:
    from core.ner_predictor import NERPredictor
    from core.scanner import DirectoryScanner
    CORE_AVAILABLE = True
except ImportError:
    CORE_AVAILABLE = False


class MLWorker(QThread):
    """
    Фоновый поток для выполнения тяжелых ML-операций.
    Изолирует инференс DistilBERT от главного GUI-потока.
    """
    log_signal = pyqtSignal(str)
    progress_signal = pyqtSignal(int)
    result_signal = pyqtSignal(list)
    finished_signal = pyqtSignal()

    def __init__(self, directory: str):
        super().__init__()
        self.directory = directory
        self.ner = None

    def run(self):
        self.log_signal.emit("[*] Инициализация интеллектуального анализа. Загрузка весов...")
        self.progress_signal.emit(10)
        
        try:
            if CORE_AVAILABLE:
                self.ner = NERPredictor()
                scanner = DirectoryScanner(self.directory, compute_hashes=False)
                inventory = scanner.scan()
            else:
                raise RuntimeError("Модули Core недоступны!")
                
            self.progress_signal.emit(40)
            
            all_files = []
            for media_list in inventory.values():
                all_files.extend(media_list)
                
            total_files = len(all_files)
            if total_files == 0:
                self.log_signal.emit("[!] В директории не найдено поддерживаемых медиафайлов.")
                self.progress_signal.emit(100)
                self.finished_signal.emit()
                return

            self.log_signal.emit(f"[*] Найдено файлов: {total_files}. Запуск нейросетевого извлечения...")
            
            results = []
            for i, media in enumerate(all_files):
                t0 = time.time()
                pred = self.ner.extract_entities(media.name)
                t_ms = (time.time() - t0) * 1000
                
                title = pred.get('title', media.name)
                year = pred.get('year', 'N/A')
                quality = pred.get('quality', 'N/A')
                
                results.append((media.name, title, year, quality, f"{t_ms:.1f} ms"))
                
                # Обновление прогресс-бара
                progress = 40 + int((i + 1) / total_files * 60)
                self.progress_signal.emit(progress)
                self.log_signal.emit(f"[NER] Индексировано: {media.name} -> {title}")
                
            self.result_signal.emit(results)
            self.log_signal.emit("[*] Обогащение метаданных успешно завершено!")
            
        except Exception as e:
            self.log_signal.emit(f"[ОШИБКА] Сбой в ML-потоке: {str(e)}")
            
        finally:
            self.progress_signal.emit(100)
            self.finished_signal.emit()


class MainWindow(QMainWindow):
    """Главное окно приложения Media Indexer."""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Интеллектуальный Индексатор Медиаконтента v1.0")
        self.setGeometry(100, 100, 1000, 700)
        self.setStyleSheet("background-color: #f4f4f9;")
        self._init_ui()

    def _init_ui(self):
        main_widget = QWidget()
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(15)

        # 1. Верхняя панель (Кнопки и статус)
        top_panel = QHBoxLayout()
        
        self.btn_select_dir = QPushButton("📁 Выбрать папку")
        self.btn_select_dir.setMinimumHeight(40)
        self.btn_select_dir.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold; border-radius: 5px;")
        self.btn_select_dir.clicked.connect(self.select_directory)
        
        self.btn_start = QPushButton("🚀 Запустить индексацию")
        self.btn_start.setMinimumHeight(40)
        self.btn_start.setStyleSheet("background-color: #2196F3; color: white; font-weight: bold; border-radius: 5px;")
        self.btn_start.clicked.connect(self.start_analysis)
        self.btn_start.setEnabled(False)

        self.lbl_status = QLabel("Ожидание выбора директории...")
        self.lbl_status.setFont(QFont("Arial", 10, QFont.Bold))
        self.lbl_status.setStyleSheet("color: #555;")

        top_panel.addWidget(self.btn_select_dir)
        top_panel.addWidget(self.btn_start)
        top_panel.addStretch()
        top_panel.addWidget(self.lbl_status)

        # 2. Прогресс-бар
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setStyleSheet("""
            QProgressBar { border: 1px solid #bbb; border-radius: 5px; text-align: center; }
            QProgressBar::chunk { background-color: #2196F3; width: 20px; }
        """)

        # 3. Таблица результатов
        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(["Исходный файл", "Извлеченное Название", "Год", "Качество", "Время (NER)"])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.table.setStyleSheet("background-color: white; gridline-color: #ccc;")

        # 4. Консоль логов
        self.console = QTextEdit()
        self.console.setReadOnly(True)
        self.console.setMaximumHeight(150)
        self.console.setStyleSheet("background-color: #1e1e1e; color: #00ff00; font-family: Consolas; font-size: 10pt;")
        self.log_to_console("Система инициализирована. Ожидание команд пользователя.")

        main_layout.addLayout(top_panel)
        main_layout.addWidget(self.progress_bar)
        main_layout.addWidget(self.table)
        main_layout.addWidget(QLabel("Системный журнал (ML Logs):"))
        main_layout.addWidget(self.console)

        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)
        
        self.target_dir = ""

    def log_to_console(self, text: str):
        time_str = time.strftime("%H:%M:%S")
        self.console.append(f"[{time_str}] {text}")
        self.console.moveCursor(QTextCursor.End)

    def select_directory(self):
        dir_path = QFileDialog.getExistingDirectory(self, "Выберите папку для индексации")
        if dir_path:
            self.target_dir = dir_path
            self.lbl_status.setText(f"Выбрана папка: {Path(dir_path).name}")
            self.log_to_console(f"Выбрана рабочая директория: {dir_path}")
            self.btn_start.setEnabled(True)

    def start_analysis(self):
        if not self.target_dir:
            return
            
        self.btn_start.setEnabled(False)
        self.btn_select_dir.setEnabled(False)
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
        for row_idx, row_data in enumerate(data):
            for col_idx, text in enumerate(row_data):
                item = QTableWidgetItem(str(text))
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                self.table.setItem(row_idx, col_idx, item)

    def on_analysis_finished(self):
        self.btn_start.setEnabled(True)
        self.btn_select_dir.setEnabled(True)
        self.lbl_status.setText("Анализ завершен.")
        QMessageBox.information(self, "Успех", "Интеллектуальная индексация файлов успешно завершена!")

def run_gui():
    app = QApplication(sys.argv)
    app.setStyle('Fusion') 
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    run_gui()