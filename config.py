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
MAX_DOWNLOADS = 100

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
TRAIN_BATCH_SIZE = 20
TRAIN_EPOCHS = 50
TRAIN_LEARNING_RATE = 1e-3
