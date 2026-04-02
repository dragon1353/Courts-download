import os

# --- 基礎網路與視窗設定 ---
FLASK_HOST = "127.0.0.1"
FLASK_PORT = 5000
WINDOW_TITLE = '地端法律 AI 分析工具 (RAG + Unsupervised)'
WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 900

# --- 司法院查詢與爬蟲設定 ---
JUDICIAL_TARGET_URL = "https://judgment.judicial.gov.tw/FJUD/Default_AD.aspx"
PDF_SAVE_PATH = r"D:\judicial_pdfs"
MAX_DOWNLOADS = 400

# --- 內部路徑設定 ---
BASE_DIR = os.path.dirname(__file__)
TEMP_DIR = os.path.join(BASE_DIR, 'temp')
COMBINED_TEXT_FILE = os.path.join(TEMP_DIR, 'combined_pdf_text.txt')
FINAL_REPORT_FILE = os.path.join(TEMP_DIR, 'final_report.html')
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
CSV_DATASET_PATH = os.path.join(DATASET_DIR, "legal_dataset.csv")
MODEL_SAVE_PATH = os.path.join(DATASET_DIR, "best_model.pth")
CHROMA_DB_DIR = os.path.join(DATASET_DIR, "chroma_db")

# --- 生成式 AI (Ollama) 整合設定 ---
OLLAMA_API_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL_NAME = "llama3.1:8b"

# --- PyTorch 模型訓練參數 (非監督式特徵學習) ---
TRAIN_MAX_VOCAB_SIZE = 10000
TRAIN_MAX_SEQ_LEN = 2048  # 建議不超過 2048 以免長序列導致顯存溢出
TRAIN_BATCH_SIZE = 128  # 經測試，16GB 顯存可穩定支援到 128，推薦以此值平衡效能與穩定性
TRAIN_EPOCHS = 50
TRAIN_LEARNING_RATE = 1e-3

# --- 自動標註關鍵字 (用於 local_auto_label.py) ---
GUILTY_KEYWORDS = [
    r"處有期徒刑", r"處拘役", r"處罰金", r"犯.*?罪", r"應執行", r"處有期", r"處刑"
]
NOT_GUILTY_KEYWORDS = [
    r"無罪", r"不受理", r"免訴", r"免刑"
]

# --- RAG 與 語意檢索參數 (動態門檻模式) ---
RAG_SIMILARITY_THRESHOLD = 0.18   # 最低相關性門檻 (高於此值才錄用)
RAG_MAX_CONTEXT_CHARS = 100000    # 大幅提升以支援無上限模式
RAG_MAX_DOC_COUNT = 15            # 提高最大參考件數

# 法律通用罪名關鍵字 (用於精準過濾)
LEGAL_CRIME_WORDS = [
    "殺人", "竊盜", "詐欺", "性剝削", "毒品", "傷害", 
    "侵佔", "侵占", "槍砲", "洗錢", "強盜", "偽造", "妨害性隱私"
]
