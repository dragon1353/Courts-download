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
    綜合所有資訊以串流方式丟給 Ollama 生成回答。
    """
    db_dir = os.path.join(config.BASE_DIR, "dataset", "chroma_db")
    
    # 預設狀態
    source_documents_html = "<p style='color: #666;'>找不到相關知識庫判例。請先執行資料下載與訓練。</p>"
    rag_context = ""
        
    # === 步驟 2: ChromaDB 知識庫檢索 ===
    try:
        if os.path.exists(db_dir):
            embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-small")
            vectorstore = Chroma(persist_directory=db_dir, embedding_function=embeddings)
            
            # 使用相似度門檻搜尋 (相似度超過 0.5 且最多抓前 20 個片段)
            retriever = vectorstore.as_retriever(
                search_type="similarity_score_threshold",
                search_kwargs={"score_threshold": 0.5, "k": 20}
            )
            docs = retriever.invoke(user_prompt)
            
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
    try:
        system_prompt = f"""
        你是一位極其專業且經驗豐富的台灣法律顧問。
        使用者提出了一個法律問題。

        【參考判例】：
        {rag_context if rag_context else "（目前知識庫中無直接相關判例）"}

        【使用者問題】：
        {user_prompt}

        【回答指令】：
        1. **優先參考**：請首先分析並引用上方提供的【參考判例】。
        2. **專業補充**：若參考判例不完整或不適用，請運用您內建的台灣法律知識與法理邏輯進行補充說明，確保回答具備專業參考價值。
        3. **誠實告知**：若該問題完全超乎您的判斷能力，請明確告知。
        4. **格式要求**：請用繁體中文，條理分明。因為輸出在網頁上，**務必使用 HTML 標籤** (如 <p>, <ul>, <li>, <strong>, <h3>)。**禁止**使用 Markdown (如 ** 或 ##)。
        """

        # 先 Yield 報告的開頭與來源部分
        header_html = f'''
        <div style="font-family: 'Segoe UI', Arial, sans-serif; line-height: 1.6; color: #333;">
            <h3 style="color: #444; border-bottom: 2px solid #ddd; padding-bottom: 5px;">📚 知識庫檢索來源</h3>
            <div style="padding: 10px 20px; border-radius: 8px; background-color: #e9ecef; border-left: 5px solid #6c757d; font-size: 0.9em; margin-bottom: 20px;">
                {source_documents_html}
            </div>
            <h3 style="color: #444; border-bottom: 2px solid #ddd; padding-bottom: 5px; margin-top: 30px;">💡 總結分析與建議 (Ollama: {config.OLLAMA_MODEL_NAME})</h3>
            <div id="streaming-content" style="padding: 20px; border-radius: 8px; background-color: #ffffff; border: 1px solid #e0e0e0; box-shadow: 0 4px 6px rgba(0,0,0,0.05);">
        '''
        yield header_html

        # 向 Ollama 發送串流請求
        req_data = json.dumps({
            "model": config.OLLAMA_MODEL_NAME,
            "prompt": system_prompt,
            "stream": True
        }).encode('utf-8')
        
        req = urllib.request.Request(
            config.OLLAMA_API_URL, 
            data=req_data, 
            headers={'Content-Type': 'application/json'}
        )
        
        with urllib.request.urlopen(req) as response:
            for line in response:
                if line:
                    chunk = json.loads(line.decode('utf-8'))
                    response_text = chunk.get("response", "")
                    if response_text:
                        yield response_text
                        if chunk.get("done", False):
                            break
        
        # 結尾 HTML
        footer_html = f'''
            </div>
            <div style="margin-top: 40px; padding: 10px; background-color: #e2e3e5; border-radius: 5px; text-align: center;">
                <p style="color: #6a6c6f; font-size: 0.85em; margin-bottom: 0;">
                    本次分析由 <b>Full Corpus RAG (ChromaDB + Gemma)</b> 驅動。所有內容皆在本地生成。
                </p>
            </div>
        </div>
        '''
        yield footer_html
            
    except Exception as e:
        yield f"<div style='color: #dc3545;'><strong>系統連線異常：</strong><br>{e}</div>"

if __name__ == "__main__":
    # 測試腳本
    for chunk in query_rag_system("測試問題"):
        print(chunk, end="", flush=True)
