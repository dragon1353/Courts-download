# --- 匯入所有需要的模組 ---
import os
import subprocess
from flask import Flask, render_template, jsonify, request, Response, send_from_directory
import pdf_loader
import local_auto_label
from rag_agent import query_rag_system # 升級為 RAG 代理
import time
import sys
import platform
import threading

# 匯入桌面應用程式視窗模組
import webview
# 匯入原生對話框模組
import tkinter as tk
from tkinter import filedialog

import config

# --- Flask 後端伺服器部分 ---
app = Flask(__name__)

# --- 配置與全域變數 ---
PDF_LOADER_SCRIPT = os.path.join(os.path.dirname(__file__), 'pdf_loader.py')
DOWNLOADER_SCRIPT = os.path.join(os.path.dirname(__file__), 'googledata.py')
TEMP_DIR = config.TEMP_DIR
COMBINED_TEXT_FILE = config.COMBINED_TEXT_FILE
FINAL_REPORT_FILE = config.FINAL_REPORT_FILE

# --- 狀態管理字典 ---
training_status = {"state": "idle", "message": "閒置"}
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

#處理PDF標註與模型訓練作業
def run_training_task(folder_path):
    global training_status
    
    # 確保設定檔裡的 PDF 路徑跟 UI 上選擇的一致
    config.PDF_SAVE_PATH = folder_path
    
    training_status = {"state": "loading", "message": "階段一：正在透過關鍵字自動標註 PDF..."}
    env = os.environ.copy()
    env['PYTHONIOENCODING'] = 'utf-8'
    
    try:
        # 第一步：執行自動標註
        auto_label_script = os.path.join(os.path.dirname(__file__), 'local_auto_label.py')
        process_label = subprocess.Popen(
            ['python', auto_label_script],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            encoding='utf-8', env=env, bufsize=1
        )
        for line in iter(process_label.stdout.readline, ''):
            line = line.strip()
            if "已處理" in line or "找到" in line or "分佈" in line:
                 training_status["message"] = f"階段一：{line}"
            print(f"[auto_label]: {line}")
        process_label.wait()

        if process_label.returncode != 0:
            training_status["state"] = "error"
            training_status["message"] = "自動標註階段發生錯誤，訓練終止。"
            return

        # 第二步：執行神經網路訓練
        training_status["message"] = "階段二：啟動 PyTorch 神經網路訓練..."
        train_model_script = os.path.join(os.path.dirname(__file__), 'train_model.py')
        process_train = subprocess.Popen(
            ['python', train_model_script],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            encoding='utf-8', env=env, bufsize=1
        )
        for line in iter(process_train.stdout.readline, ''):
            line = line.strip()
            if "Epoch" in line:
                 training_status["message"] = f"正在訓練神經網路... {line}"
            elif "已儲存" in line:
                 training_status["message"] = f"模型儲存成功：{line}"
            print(f"[train_model]: {line}")
            
        process_train.wait()
        
        if process_train.returncode == 0:
            training_status["state"] = "success"
            training_status["message"] = "煉丹完成！已成功儲存最佳模型 best_model.pth。"
        else:
             training_status["state"] = "error"
             training_status["message"] = f"神經網路訓練階段發生錯誤，返回碼 {process_train.returncode}。"

    except Exception as e:
        training_status["state"] = "error"
        training_status["message"] = f"啟動訓練任務時發生錯誤: {e}"

#處理 RAG 問答與分析作業
def run_analysis_task(user_prompt):
    global analysis_status
    analysis_status = {"state": "analyzing", "message": "正在處理您的提問，請稍候..."}
    try:
        print(f"啟動 RAG 分析，User Prompt: {user_prompt}")
        # 呼叫新的 RAG Agent
        html_report = query_rag_system(user_prompt)

        analysis_status["state"] = "success"
        analysis_status["message"] = "分析完成"
        # 將生成的 HTML 放進狀態中傳給前端
        analysis_status["report_html"] = html_report
    except Exception as e:
        analysis_status["state"] = "error"
        analysis_status["message"] = f"分析過程發生錯誤: {e}"

# --- Flask 路由 ---
@app.route('/')
def index():
    return render_template('index.html')


# --- API 端點 ---
@app.route('/start_download', methods=['POST'])
def start_download_api():
    if training_status["state"] == "loading" or analysis_status["state"] == "analyzing" or download_status["state"] == "downloading":
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


@app.route('/start_training', methods=['POST'])
def start_training_api():
    if training_status["state"] == "loading" or analysis_status["state"] == "analyzing" or download_status["state"] == "downloading":
        return jsonify({"status": "error", "message": "已有任務在運行中。"}), 409
    folder_path = request.get_json().get('folder_path')
    if not folder_path: return jsonify({"status": "error", "message": "未提供資料夾路徑。"}), 400
    thread = threading.Thread(target=run_training_task, args=(folder_path,)); thread.daemon = True; thread.start()
    return jsonify({"status": "success"})

@app.route('/training_status')
def get_training_status_api():
    return jsonify(training_status)

@app.route('/start_analysis', methods=['POST'])
def start_analysis_api():
    if training_status["state"] == "loading" or analysis_status["state"] == "analyzing" or download_status["state"] == "downloading":
        return jsonify({"status": "error", "message": "已有任務在運行中。"}), 409
    
    user_prompt = request.get_json().get('user_prompt')
    if not user_prompt: 
        return jsonify({"status": "error", "message": "請輸入提問或分析指令。"}), 400
        
    thread = threading.Thread(target=run_analysis_task, args=(user_prompt,))
    thread.daemon = True
    thread.start()
    return jsonify({"status": "success", "message": "分析任務已啟動"})

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