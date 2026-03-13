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
MAX_DOWNLOADS = 25

# --- 內部路徑設定 ---
BASE_DIR = os.path.dirname(__file__)
TEMP_DIR = os.path.join(BASE_DIR, 'temp')
COMBINED_TEXT_FILE = os.path.join(TEMP_DIR, 'combined_pdf_text.txt')
FINAL_REPORT_FILE = os.path.join(TEMP_DIR, 'final_report.html')
