import os
import vertexai
from vertexai.generative_models import GenerativeModel
import datetime
import config

# --- 設定你的 Google Cloud 專案資訊 ---
PROJECT_ID = config.GEMINI_PROJECT_ID
LOCATION = config.GEMINI_LOCATION

# --- 設定服務帳戶金鑰路徑 ---
KEY_FILE_PATH = config.GEMINI_KEY_FILE_PATH

# 設定 GOOGLE_APPLICATION_CREDENTIALS 環境變數
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = KEY_FILE_PATH

# 初始化 Vertex AI
try:
    vertexai.init(project=PROJECT_ID, location=LOCATION)
    print("Vertex AI 初始化成功！")
except Exception as e:
    print(f"Vertex AI 初始化失敗：{e}")
    # 在實際應用中，這裡應該有更完善的錯誤處理
    
def generate_analysis_report(pdf_content, user_prompt):
    """
    接收 PDF 文字內容和使用者指令，呼叫 Gemini AI 生成分析報告。
    
    Returns:
        str: HTML 格式的報告。
    """
    print("接收到分析請求，正在呼叫 Gemini API...")
    
    try:
        model = GenerativeModel(config.GEMINI_MODEL_NAME)
        
        # 建立一個強大的 Prompt
        prompt = f"""
        你是一位專業的數據分析師。這裡有多份文件的合併文字內容，以及一個特定的使用者指令。
        請根據所有提供的文件內容，全面地回應使用者的指令。你的回覆應該結構清晰、內容詳盡。

        --- 文件合併內容 (可能包含多份文件) ---
        {pdf_content[:20000]}  # 限制長度以符合模型單次請求限制，可根據模型調整

        --- 使用者指令 ---
        {user_prompt}

        --- 你的分析報告 ---
        請直接開始撰寫你的分析報告。
        """

        response = model.generate_content(prompt)
        analysis_text = response.text

    except Exception as e:
        print(f"Gemini API 呼叫失敗: {e}")
        analysis_text = f"<h2 style='color:red;'>分析時發生錯誤</h2><p>無法完成分析請求，錯誤細節如下：</p><pre>{e}</pre>"

    # 將模型的純文字回覆包裝成美觀的 HTML
    import html
    escaped_prompt = html.escape(user_prompt)
    
    html_report = f"""
    <!DOCTYPE html>
    <html lang="zh-TW">
    <head>
        <meta charset="UTF-8">
        <style>
            body {{ font-family: 'Microsoft JhengHei', sans-serif; line-height: 1.8; padding: 15px; color: #333; }}
            h1, h2 {{ color: #0056b3; border-bottom: 1px solid #ddd; padding-bottom: 5px; }}
            .query {{ background-color: #eef; padding: 10px; border-left: 4px solid #0056b3; margin-bottom: 20px; }}
            .response {{ white-space: pre-wrap; word-wrap: break-word; }}
            .footer {{ margin-top: 20px; font-size: 0.8em; color: #888; text-align: center; }}
        </style>
    </head>
    <body>
        <h1>Gemini AI 判決書分析報告</h1>
        <div class="query">
            <h2>使用者分析指令</h2>
            <p>{escaped_prompt}</p>
        </div>
        
        <h2>分析結果</h2>
        <div class="response">
            {analysis_text.replace(os.linesep, '<br>')}
        </div>

        <div class="footer">報告生成時間: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>
    </body>
    </html>
    """
    print("報告生成完畢。")
    return html_report