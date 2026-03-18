import os
import pathlib

# --- 路徑設定 (Paths) ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 原始資料來源 (Triage)
SOURCE_TILE_DIR_FRONT = os.path.join(BASE_DIR, 'Triage', 'Front_Dataset')
SOURCE_TILE_DIR_BACK = os.path.join(BASE_DIR, 'Triage', 'Back_Dataset')

# 訓練/測試資料輸出 (Test)
DATA_ROOT = os.path.join(BASE_DIR, 'Test')
OUTPUT_SPLIT_DIR_FRONT = os.path.join(DATA_ROOT, "Front_Dataset")
OUTPUT_SPLIT_DIR_BACK = os.path.join(DATA_ROOT, "Back_Dataset")

# 網頁上傳暫存
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static', 'uploads')

# 模型與門檻檔案路徑
MODEL_PATH_FRONT = os.path.join(BASE_DIR, 'front_mil.pth')
MODEL_PATH_BACK = os.path.join(BASE_DIR, 'back_mil.pth')
THRESHOLD_PATH_FRONT = os.path.join(BASE_DIR, 'front_threshold.json')
THRESHOLD_PATH_BACK = os.path.join(BASE_DIR, 'back_threshold.json')
HISTORY_IMG_PATH_FRONT = os.path.join(BASE_DIR, 'front_autoencoder_history.png')
HISTORY_IMG_PATH_BACK = os.path.join(BASE_DIR, 'back_autoencoder_history.png')

# --- 圖片與模型參數 (Image & Model Params) ---
IMG_HEIGHT = 224
IMG_WIDTH = 224
IMG_CHANNELS = 3
LATENT_DIM = 64

# --- 訓練參數 (Training Params) ---
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.00001
VALIDATION_SPLIT = 0.2
RANDOM_SEED = 42

# --- 網頁設定 (Web App Params) ---
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp'}
MAX_CONTENT_LENGTH = 4 * 1024 * 1024 * 1024  # 4 GB

# --- MIL Parameters ---
MAX_TILES = 128
