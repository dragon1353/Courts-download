# --- 匯入所有需要的模組 ---
import os
import subprocess
# --- 修改：從 Flask 匯入 render_template ---
from flask import Flask, render_template, jsonify, request, Response, send_from_directory
import threading
import datetime
import time
import sys
import platform

# 匯入桌面應用程式視窗模組
import webview
# 匯入原生對話框模組
import tkinter as tk
from tkinter import filedialog

# 匯入我們自己的分析器
import gemini_analyzer
import config

# --- Flask 後端伺服器部分 ---
app = Flask(__name__)

# --- 配置與全域變數 ---
PDF_LOADER_SCRIPT = os.path.join(os.path.dirname(__file__), 'pdf_loader.py')
DOWNLOADER_SCRIPT = os.path.join(os.path.dirname(__file__), 'googledata.py')
TEMP_DIR = config.TEMP_DIR
COMBINED_TEXT_FILE = config.COMBINED_TEXT_FILE
FINAL_REPORT_FILE = config.FINAL_REPORT_FILE

# --- 狀態管理字典 (新增 download_status) ---
loading_status = {"state": "idle", "message": "閒置"}
analysis_status = {"state": "idle", "message": "閒置"}
download_status = {"state": "idle", "message": "閒置"}

# --- 背景任務 ---

def run_download_task(start_y, start_m, start_d, end_y, end_m, end_d):
    """背景任務：非同步執行 googledata.py 並更新 download_status"""
    global download_status
    download_status = {"state": "downloading", "message": "正在初始化下載任務..."}

    env = os.environ.copy()
    env['PYTHONIOENCODING'] = 'utf-8'

    try:
        process = subprocess.Popen(
            ['python', DOWNLOADER_SCRIPT, start_y, start_m, start_d, end_y, end_m, end_d],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            encoding='utf-8', env=env, bufsize=1
        )
        for line in iter(process.stdout.readline, ''):
            line = line.strip()
            download_status["message"] = line
            print(f"[googledata]: {line}")

        process.wait()
        return_code = process.returncode
        if return_code == 0:
            download_status["state"] = "success"
            download_status["message"] = "PDF 下載流程執行完畢。"
        else:
            download_status["state"] = "error"
            download_status["message"] = f"下載腳本執行出錯，返回碼: {return_code}"

    except FileNotFoundError:
        download_status["state"] = "error"
        download_status["message"] = f"錯誤：找不到腳本 {DOWNLOADER_SCRIPT}"
    except Exception as e:
        download_status["state"] = "error"
        download_status["message"] = f"啟動下載腳本時發生錯誤: {e}"

#處理PDF讀取至暫存作業
def run_loading_task(folder_path):
    global loading_status
    print("正在清除舊的暫存檔案...")
    try:
        if os.path.exists(COMBINED_TEXT_FILE): os.remove(COMBINED_TEXT_FILE)
        if os.path.exists(FINAL_REPORT_FILE): os.remove(FINAL_REPORT_FILE)
    except Exception as e:
        print(f"清除舊檔案時發生錯誤: {e}")
    loading_status = {"state": "loading", "message": "正在初始化讀取任務..."}
    env = os.environ.copy()
    env['PYTHONIOENCODING'] = 'utf-8'
    try:
        process = subprocess.Popen(
            ['python', PDF_LOADER_SCRIPT, folder_path],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            encoding='utf-8', env=env, bufsize=1
        )
        for line in iter(process.stdout.readline, ''):
            line = line.strip()
            if line.startswith("PROGRESS:"):
                loading_status["message"] = f"正在讀取檔案... ({line.split(':', 1)[1]})"
            elif line.startswith("FINAL_SUCCESS:"):
                loading_status["state"] = "success"
                loading_status["message"] = line.split(":", 1)[1]
            elif line.startswith("FINAL_ERROR:"):
                 loading_status["state"] = "error"
                 loading_status["message"] = line.split(":", 1)[1]
            print(f"[pdf_loader]: {line}")
        process.wait()
        if loading_status["state"] == "loading":
            loading_status["state"] = "error"
            loading_status["message"] = "讀取腳本意外終止。"
    except Exception as e:
        loading_status["state"] = "error"
        loading_status["message"] = f"啟動讀取腳本時發生錯誤: {e}"
#處理
def run_analysis_task(user_prompt):
    global analysis_status
    analysis_status = {"state": "analyzing", "message": "正在讀取已上傳的資料..."}
    try:
        with open(COMBINED_TEXT_FILE, 'r', encoding='utf-8') as f:
            pdf_content = f.read()
        analysis_status["message"] = "資料讀取完畢，正在發送至 Gemini AI 進行分析..."
        html_report = gemini_analyzer.generate_analysis_report(pdf_content, user_prompt)
        with open(FINAL_REPORT_FILE, 'w', encoding='utf-8') as f:
            f.write(html_report)
        analysis_status["state"] = "success"
        analysis_status["message"] = "分析完成！報告已生成。"
        analysis_status["report_html"] = html_report
    except FileNotFoundError:
        analysis_status["state"] = "error"
        analysis_status["message"] = "錯誤：找不到暫存的 PDF 資料檔，請先執行『資料上傳』。"
    except Exception as e:
        analysis_status["state"] = "error"
        analysis_status["message"] = f"分析過程中發生未知錯誤: {e}"

# --- Flask 路由 ---
@app.route('/')
def index():
    # --- 修改：渲染獨立的 index.html 模板 ---
    return render_template('index.html')


# --- API 端點 ---
@app.route('/start_download', methods=['POST'])
def start_download_api():
    if loading_status["state"] == "loading" or analysis_status["state"] == "analyzing" or download_status["state"] == "downloading":
        return jsonify({"status": "error", "message": "已有任務在運行中。"}), 409

    data = request.get_json()
    if not data:
        return jsonify({"status": "error", "message": "未提供日期資料。"}), 400

    start_y = data.get('start_year')
    start_m = data.get('start_month')
    start_d = data.get('start_day')
    end_y = data.get('end_year')
    end_m = data.get('end_month')
    end_d = data.get('end_day')

    if not all([start_y, start_m, start_d, end_y, end_m, end_d]):
        return jsonify({"status": "error", "message": "日期參數不完整。"}), 400

    thread = threading.Thread(target=run_download_task, args=(start_y, start_m, start_d, end_y, end_m, end_d))
    thread.daemon = True
    thread.start()
    return jsonify({"status": "success"})

@app.route('/download_status')
def get_download_status_api():
    return jsonify(download_status)


@app.route('/start_loading', methods=['POST'])
def start_loading_api():
    if loading_status["state"] == "loading" or analysis_status["state"] == "analyzing" or download_status["state"] == "downloading":
        return jsonify({"status": "error", "message": "已有任務在運行中。"}), 409
    folder_path = request.get_json().get('folder_path')
    if not folder_path: return jsonify({"status": "error", "message": "未提供資料夾路徑。"}), 400
    thread = threading.Thread(target=run_loading_task, args=(folder_path,)); thread.daemon = True; thread.start()
    return jsonify({"status": "success"})

@app.route('/loading_status')
def get_loading_status_api():
    return jsonify(loading_status)

@app.route('/start_analysis', methods=['POST'])
def start_analysis_api():
    if loading_status["state"] == "loading" or analysis_status["state"] == "analyzing" or download_status["state"] == "downloading":
        return jsonify({"status": "error", "message": "已有任務在運行中。"}), 409
    if not os.path.exists(COMBINED_TEXT_FILE): return jsonify({"status": "error", "message": "請先『資料上傳』成功後再執行分析。"}), 400
    user_prompt = request.get_json().get('user_prompt')
    if not user_prompt: return jsonify({"status": "error", "message": "未提供分析指令。"}), 400
    thread = threading.Thread(target=run_analysis_task, args=(user_prompt,)); thread.daemon = True; thread.start()
    return jsonify({"status": "success", "message": "分析任務已啟動。"})

@app.route('/analysis_status')
def get_analysis_status_api():
    return jsonify(analysis_status)

@app.route('/get_final_report')
def get_final_report_api():
    return send_from_directory(TEMP_DIR, os.path.basename(FINAL_REPORT_FILE))


# --- pywebview 桌面應用程式啟動器 ---
class Api:
    def select_folder(self):
        root = tk.Tk()
        root.withdraw()
        folder = filedialog.askdirectory(title="請選擇包含 PDF 的資料夾")
        root.destroy()
        return folder

def run_flask():
    app.run(host=config.FLASK_HOST, port=config.FLASK_PORT)

if __name__ == '__main__':
    os.makedirs(TEMP_DIR, exist_ok=True)
    api = Api()
    flask_thread = threading.Thread(target=run_flask)
    flask_thread.daemon = True
    flask_thread.start()
    webview.create_window(
        config.WINDOW_TITLE,
        f'http://{config.FLASK_HOST}:{config.FLASK_PORT}',
        js_api=api,
        width=config.WINDOW_WIDTH,
        height=config.WINDOW_HEIGHT
    )
    webview.start()