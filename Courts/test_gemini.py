import os
import sys

# 將當前目錄加入搜尋路徑
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import config
import vertexai
from vertexai.generative_models import GenerativeModel

print(f"專案 ID (Project ID): {config.GEMINI_PROJECT_ID}")
print(f"區域 (Location): {config.GEMINI_LOCATION}")
print(f"金鑰路徑 (Key Path): {config.GEMINI_KEY_FILE_PATH}")
print("-" * 30)

if not os.path.exists(config.GEMINI_KEY_FILE_PATH):
    print(f"錯誤：找不到金鑰檔案於 {config.GEMINI_KEY_FILE_PATH}！")
    sys.exit(1)

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = config.GEMINI_KEY_FILE_PATH

try:
    print("嘗試初始化 Vertex AI...")
    vertexai.init(project=config.GEMINI_PROJECT_ID, location=config.GEMINI_LOCATION)
    print("Vertex AI 初始化成功！\n")
except Exception as e:
    print(f"Vertex AI 初始化失敗：{e}")
    sys.exit(1)

models_to_test = [
    "gemini-2.5-pro",
    "gemini-3.1-pro-preview"
]

for model_name in models_to_test:
    print(f"==================================================")
    print(f"測試模型: {model_name}")
    try:
        model = GenerativeModel(model_name)
        response = model.generate_content("這是一個測試訊息，如果你收到了請回覆「連線成功」。")
        print(f"✅ 狀態: 成功")
        print(f"回覆內容: {response.text.strip()}")
    except Exception as e:
        error_msg = str(e)
        if "404" in error_msg and "not found" in error_msg.lower():
            print(f"❌ 狀態: 失敗 - 目前 GCP 不開放此模型版本 (404 Not Found)")
        elif "403" in error_msg:
            print(f"❌ 狀態: 失敗 - 權限不足或未啟用 API (403 Permission Denied)")
        else:
            print(f"❌ 狀態: 失敗 - 其他錯誤 ({error_msg[:100]}...)")
    print(f"==================================================\n")
