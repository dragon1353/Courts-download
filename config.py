import os

# --- 基礎網路設定 ---
FLASK_HOST = "127.0.0.1"
FLASK_PORT = 5000

# --- 視窗應用程式設定 ---
WINDOW_TITLE = 'PDF 分析工具'
WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 900

# --- Gemini API 設定 ---
GEMINI_PROJECT_ID = "aicoding-463201"
GEMINI_LOCATION = "us-central1"
GEMINI_KEY_FILE_PATH = r"E:\Germini API\aicoding-463201-2ad2d415de6b.json"
GEMINI_MODEL_NAME = "gemini-2.5-pro"

# --- 司法院查詢與爬蟲設定 ---
JUDICIAL_TARGET_URL = "https://judgment.judicial.gov.tw/FJUD/Default_AD.aspx"
PDF_SAVE_PATH = r"D:\judicial_pdfs"
MAX_DOWNLOADS = 100

# --- 內部路徑設定 ---
BASE_DIR = os.path.dirname(__file__)
TEMP_DIR = os.path.join(BASE_DIR, 'temp')
COMBINED_TEXT_FILE = os.path.join(TEMP_DIR, 'combined_pdf_text.txt')
FINAL_REPORT_FILE = os.path.join(TEMP_DIR, 'final_report.html')

# --- 機器學習與資料集設定 ---
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
CSV_DATASET_PATH = os.path.join(DATASET_DIR, "legal_dataset.csv")
MODEL_SAVE_PATH = os.path.join(DATASET_DIR, "best_model.pth")

# --- 生成式 AI (Ollama) 整合設定 ---
OLLAMA_API_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL_NAME = "gemma3:12b"

# --- 本地自動標註關鍵字設定 ---
GUILTY_KEYWORDS = [r"處有期徒刑", r"科罰金", r"處拘役", r"沒收", r"宣告緩刑"]
NOT_GUILTY_KEYWORDS = [r"無罪", r"公訴不受理", r"免訴", r"管轄錯誤"]

# --- PyTorch 模型訓練參數 ---
TRAIN_MAX_VOCAB_SIZE = 8000
TRAIN_MAX_SEQ_LEN = 800
TRAIN_BATCH_SIZE = 16
TRAIN_EPOCHS = 5
TRAIN_LEARNING_RATE = 1e-3
FOCAL_LOSS_ALPHA = 0.25
FOCAL_LOSS_GAMMA = 2.0
