import torch
import os
import sys
import json
import urllib.request
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# 確保載入 config
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config
# PyTorch 特徵驗證已依需求捨棄，專注於全局 RAG 檢索
def query_rag_system(user_prompt):
    """
    接收使用者的全局問題，去 ChromaDB 檢索相關判例。
    綜合所有資訊丟給 Ollama 生成回答。
    """
    db_dir = os.path.join(config.BASE_DIR, "dataset", "chroma_db")
    
    # 預設狀態
    source_documents_html = "<p style='color: #666;'>找不到相關知識庫判例。請先執行資料上傳。</p>"
    rag_context = ""
        
    # === 步驟 2: ChromaDB 知識庫檢索 ===
    try:
        if os.path.exists(db_dir):
            embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-small")
            vectorstore = Chroma(persist_directory=db_dir, embedding_function=embeddings)
            
            # 使用者的問題作為搜尋 Query，找出最接近的 5 個片段
            docs = vectorstore.similarity_search(user_prompt, k=5)
            
            if docs:
                unique_filenames = set()
                for i, doc in enumerate(docs):
                    source = doc.metadata.get("source", "未知來源")
                    filename = os.path.basename(source)
                    unique_filenames.add(filename)
                    rag_context += f"[參考資料 {i+1} - {filename}]:\n{doc.page_content}\n\n"
                
                source_documents_html = "<ul>"
                for filename in unique_filenames:
                    source_documents_html += f"<li><strong>{filename}</strong></li>"
                source_documents_html += "</ul>"
    except Exception as e:
        source_documents_html = f"<span style='color: red;'>檢索異常: {e}</span>"

    # === 步驟 3: 組合 Prompt 給 Ollama (Gemma) ===
    gemma_response_html = "<p>正在生成分析報告，但未收到內容...</p>"
    try:
        system_prompt = f"""
        你是一位專業的台灣法律顧問。
        使用者提出了一個問題。請你**只根據下面提供的【參考判例】**來回答使用者的問題。
        如果參考判例不足以回答，請明確告知「根據目前的知識庫資料不足以判斷」。
        
        【參考判例】：
        {rag_context if rag_context else "無參考資料"}
        
        【使用者問題】：
        {user_prompt}
        
        請用繁體中文，以專業、清晰、條理分明的方式回答。因為你要輸出在網頁上，所以**請務必使用 HTML 標籤排版** (例如 <p>, <ul>, <li>, <strong>, <h3>)。**絕對不要**使用 Markdown 語法 (如 ** 或 ##)。
        """

        req_data = json.dumps({
            "model": config.OLLAMA_MODEL_NAME,
            "prompt": system_prompt,
            "stream": False
        }).encode('utf-8')
        
        req = urllib.request.Request(
            config.OLLAMA_API_URL, 
            data=req_data, 
            headers={'Content-Type': 'application/json'}
        )
        
        with urllib.request.urlopen(req) as response:
            result = json.loads(response.read().decode('utf-8'))
            gemma_response_html = result.get("response", "<p>未回傳有效內容</p>")
            
    except Exception as e:
        gemma_response_html = f"<div style='color: #dc3545;'><strong>Ollama 連線失敗：</strong><br>無法連接到您的地端大模型 ({config.OLLAMA_MODEL_NAME})。錯誤訊息：{e}<br>請確認 Ollama 正在運行。</div>"

    # === 步驟 4: 組合最終 HTML 報表 ===
    html_report = f'''
    <div style="font-family: 'Segoe UI', Arial, sans-serif; line-height: 1.6; color: #333;">
        
        <!-- RAG 檢索來源區塊 -->
        <h3 style="color: #444; border-bottom: 2px solid #ddd; padding-bottom: 5px;">📚 知識庫檢索來源 (Top 5 相關判例)</h3>
        <div style="padding: 10px 20px; border-radius: 8px; background-color: #e9ecef; border-left: 5px solid #6c757d; font-size: 0.9em; margin-bottom: 20px;">
            {source_documents_html}
        </div>
    '''

    html_report += f'''
        <!-- Gemma 生成分析區塊 -->
        <h3 style="color: #444; border-bottom: 2px solid #ddd; padding-bottom: 5px; margin-top: 30px;">💡 總結分析與建議 (Ollama: {config.OLLAMA_MODEL_NAME})</h3>
        <div style="padding: 20px; border-radius: 8px; background-color: #ffffff; border: 1px solid #e0e0e0; box-shadow: 0 4px 6px rgba(0,0,0,0.05);">
            {gemma_response_html}
        </div>
        
        <div style="margin-top: 40px; padding: 10px; background-color: #e2e3e5; border-radius: 5px; text-align: center;">
            <p style="color: #6a6c6f; font-size: 0.85em; margin-bottom: 0;">
                本次分析由 <b>Full Corpus RAG (ChromaDB + Gemma)</b> 驅動，針對您的全體判決書資料庫進行檢索與推論。全程離線運算保障隱私。
            </p>
        </div>
    </div>
    '''
    return html_report

if __name__ == "__main__":
    # 測試腳本
    print("測試 RAG 系統...")
    print(query_rag_system("這50份判決書中，關於竊盜罪的判刑狀況為何？"))
