import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import os
import sys
import json
import time
import urllib.request
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- [Google TurboQuant 整合] ---
# 已將庫遷移至專案目錄下的 turboquant_pkg 以解決相對路徑導入問題
try:
    from turboquant_pkg.turboquant import TurboQuantProd
    # 初始化 256 維 (Autoencoder 輸出維度), 4-bit 量化引擎
    tq_engine = TurboQuantProd(d=256, bits=4, device="cpu")
    print("🚀 [TurboQuant] 4-bit 量化引擎載入成功")
except Exception as e:
    tq_engine = None
    print(f"⚠️ [TurboQuant] 載入失敗: {e}")
from models import Vocab, LegalAutoencoder

# 確保載入 config
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config

# 標準化函式與法律用語同義詞對照 (Global Utility)
def normalize_legal(text):
    if not text or not isinstance(text, str): return ""
    # 1. 基礎簡繁與異體字轉換
    t = text.replace("佔", "占").replace("臺", "台").replace("妳", "你")
    
    # 2. 法律術語同義詞對照 (Synonym Mapping)
    # 將常用口語轉換為法律正式用語，確保 TF-IDF 與硬性過濾能精準對接
    synonyms = {
        "詐騙": "詐欺",
        "騙子": "詐欺",
        "殺": "殺人",
        "偷": "竊盜",
        "拿": "侵占",
        "侵佔": "侵占",
        "性侵": "強制性交",
        "強姦": "強制性交",
        "毒品": "毒品",
        "車禍": "過失傷害", "撞到": "交通", "賠償": "侵權行為"
    }
    for k, v in synonyms.items():
        if k in t:
            t = t.replace(k, v)
    return t.strip().lower()

# 全域變數載入模型
_model = None
_vocab = None
_model_features_cache = {} 
# [優化] 強制將特徵提取模型放在 CPU，避免與 Ollama 搶奪 GPU 顯存導致分析卡死 (死鎖)
_device = torch.device("cpu")

def load_feature_model():
    global _model, _vocab
    model_path = os.path.join(config.BASE_DIR, "dataset", "best_model.pth")
    if os.path.exists(model_path):
        try:
            # 確保是在正確的裝置上載入
            checkpoint = torch.load(model_path, map_location=_device)
            _vocab = Vocab(checkpoint['vocab'], checkpoint['inv_vocab'])
            _model = LegalAutoencoder(len(_vocab.vocab))
            _model.load_state_dict(checkpoint['model_state_dict'])
            _model.to(_device)
            _model.eval()
            print(f"成功載入【完全地端】法律特徵模型 (Device: {_device})。")
        except Exception as e:
            print(f"載入特徵模型失敗: {e}")

def extract_legal_search_terms(query):
    """
    [智囊模式]: 使用 Llama 3.1 8B 解析使用者的口語問題，並轉換為專業法律檢索詞。
    """
    import requests
    import json
    
    # 建立精簡的 Prompt 請求 Llama 產出關鍵字
    prompt = f"請將以下口語化的法律問題，轉化為 3-5 個專業的法律檢索關鍵字（例如：案由、法條、行為關鍵字）。只需回覆關鍵字並以空白分隔，其餘廢話都不要。問題：{query}"
    
    payload = {
        "model": config.OLLAMA_MODEL_NAME, # 修正變數名
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.3, "num_predict": 30}
    }
    
    try:
        response = requests.post(config.OLLAMA_API_URL, json=payload, timeout=300) # 增加超時時間至 300 秒 (5 分鐘)
        if response.status_code == 200:
            result = response.json().get("response", "").strip()
            # 清理可能的多餘符號
            cleaned_result = result.replace("、", " ").replace(",", " ").replace("：", "")
            print(f"🧠 [AI 智慧解析關鍵字]: {cleaned_result}")
            return cleaned_result
    except Exception as e:
        print(f"⚠️ AI 關鍵字解析失敗: {e}")
    
    return query # 若失敗則退而求其次使用原句

def get_latent_features(text):
    if _model is None or _vocab is None: return None
    try:
        indices = _vocab.encode(text, config.TRAIN_MAX_SEQ_LEN)
        input_tensor = torch.tensor([indices], dtype=torch.long).to(_device)
        with torch.no_grad():
            _, latent = _model(input_tensor)
        return latent.cpu().numpy()[0]
    except:
        return None

# 初始化
load_feature_model()

class LocalLegalEmbedding(torch.nn.Module):
    # 這裡為了符合 langchain 介面，簡單封裝
    def embed_query(self, text):
        return get_latent_features(text).tolist()
    def embed_documents(self, texts):
        return [self.embed_query(t) for t in texts]

def perform_statistical_analysis(query):
    """
    從 legal_stats.csv 提取統計資訊，提供全局數據。
    【動態模式】：自動感應資料庫中所有罪名，不寫死。
    """
    stats_path = os.path.join(config.DATASET_DIR, "legal_stats.csv")
    if not os.path.exists(stats_path):
        return None
    
    try:
        df = pd.read_csv(stats_path)
        total_count = len(df)
        
        # 1. 動態取得資料庫現有罪名清單 (排除未知)
        unique_crimes = [c for c in df['CrimeType'].unique() if c and c != "未知"]
        
        # 2. 定義統計觸發詞 (這部分為語意基礎，較固定)
        quantity_keywords = ["多少", "數量", "份數", "總數", "總共", "幾件", "平均", "比例", "統計", "分析"]
        is_stat_query = any(k in query for k in quantity_keywords)
        
        # 3. 動態識別使用者想問哪種罪名
        target_crime = None
        for crime in unique_crimes:
            if crime in query:
                target_crime = crime
                break
        
        # 若兩者都沒對到，則視為非統計題
        if not is_stat_query and not target_crime:
            return None

        return {
            "is_stat_query": is_stat_query,
            "global_total": total_count,
            "target_crime": target_crime,
            "crime_stats": None,
            "top_crimes": df['CrimeType'].value_counts().head(5).to_dict(),
            "global_avg_sentence": round(df['SentenceMonths'].mean(), 1)
        } if (is_stat_query or target_crime) else {
            "is_stat_query": False,
            "global_total": total_count,
            "target_crime": None,
            "crime_stats": None,
            "top_crimes": {},
            "global_avg_sentence": 0
        }
    except Exception as e:
        print(f"統計分析執行失敗: {e}")
        return None

def query_rag_system(user_prompt):
    """
    接收使用者的全局問題，去 ChromaDB 檢索相關判例。
    綜合所有資訊以串流方式丟給 Ollama 生成回答。
    """
    db_dir = os.path.join(config.BASE_DIR, "dataset", "chroma_db")
    
    # 預設狀態
    source_documents_html = "<p style='color: #666;'>找不到相關知識庫判例。請先執行資料下載與訓練。</p>"
    rag_context = ""
        
    # === 步驟 0: 結構化統計分析 ===
    analysis_data = perform_statistical_analysis(user_prompt)
    stats_html = ""
    stats_text_for_ai = "（本次問題未觸發特定統計模式）"

    if analysis_data and analysis_data.get("global_total", 0) > 0:
        total = analysis_data["global_total"]
        top_crimes_dict = analysis_data.get("top_crimes", {})
        top_crimes_str = "、".join([f"{k}({v}件)" for k, v in top_crimes_dict.items()])
        
        # 建立 AI 用的純文字 (作為背景知識，不論是否顯示 UI 都給 AI)
        stats_text_for_ai = f"1. 目前資料庫累積判決總數：{total} 件\n"
        if top_crimes_str:
            stats_text_for_ai += f"2. 主要案件類型分佈：{top_crimes_str}\n"

        if analysis_data.get("target_crime"):
            # 只有在特定的統計查詢或識別到明確罪名時，才在 UI 顯示統計框
            if analysis_data.get("is_stat_query") or analysis_data.get("target_crime"):
                stats_html = '<div style="background-color: #fff3cd; border: 1px solid #ffeeba; padding: 15px; border-radius: 8px; margin-bottom: 20px; color: #856404;">'
                stats_html += f"📊 **地端數據庫全局快照**：<br>● **已索引判決總數**：{total} 件<br>"
                
                # 若有目標罪名的詳細統計，也補上
                # (這裡因為簡化，我們只顯示總量與類型，若需更細則可再擴充)
                stats_html += f"● **偵測到相關類別**：{analysis_data['target_crime']}<br>"
                stats_html += "</div>"
        
            stats_text_for_ai += f"3. 關於使用者詢問的類別：{analysis_data['target_crime']}\n"
    
    # 若非統計問題，則 stats_html 保持空字串，不干擾場景題 UI

    # === 步驟 1: AI 智慧關鍵字擴張 (Agentic Search) ===
    # [核心優化] 增加 16KB 的填充，強迫跨越所有網路代理層與瀏覽器緩衝區 (確保即時渲染，請勿縮減此處)
    padding = " " * 16384
    yield f'''
    <style>
        @keyframes pulse {{ 0% {{ opacity: 0.5; }} 50% {{ opacity: 1; }} 100% {{ opacity: 0.5; }} }}
        .pulse {{ animation: pulse 1.5s infinite; }}
        .step-done {{ color: #28a745; font-weight: bold; margin-top:5px; }}
        .step-active {{ color: #007bff; font-weight: bold; margin-top:5px; }}
        .step-pending {{ color: #999; margin-top:5px; }}
    </style>
    <div id="progress-indicator" style="padding:20px; background-color:#f8f9fa; border-radius:12px; margin-bottom:25px; border:1px solid #dee2e6; font-family: sans-serif; box-shadow: 0 2px 4px rgba(0,0,0,0.05);">
        <p style="margin:0 0 15px 0; border-bottom:2px solid #eee; padding-bottom:10px; font-size:1.1em;">📋 <b>地端 RAG 執行軌跡 (AI 優先模式)</b></p>
        <div id="step-1" class="step-active pulse">● 階段 1: 正在精準解析提問意圖與法律關鍵字...</div>
        <div id="step-2" class="step-pending">○ 階段 2: 正在生成全量深度文字矩陣 (N-gram 1-5)...</div>
        <div id="step-3" class="step-pending">○ 階段 3: 正在進行全庫相似度掃描與候選召回...</div>
        <div id="step-4" class="step-pending">○ 階段 4: 正在進行本地模型語意重排 (Reranking)...</div>
        <div id="step-5" class="step-pending">○ 階段 5: 正在進行 Llama 總結分析與法律建議產出...</div>
    </div>
    ''' + padding
    
    ai_search_keywords = extract_legal_search_terms(user_prompt)
    search_query = f"{user_prompt} {ai_search_keywords}"
    
    # 更新步驟 1 為完成，啟動步驟 2 (將 script 合併為一筆以防 chunk 斷開導致 raw JS 顯示)
    yield f'<script>document.getElementById("step-1").innerHTML = "✅ 階段 1: 法律解析完成：{ai_search_keywords}"; document.getElementById("step-1").className = "step-done"; document.getElementById("step-2").className = "step-active pulse";</script>'
    yield padding
    time.sleep(0.5) 

    # === 步驟 2: PyTorch 特徵提取 (CPU 模式) ===
    print("🔍 [Heartbeat] 正在活化地端向量模型 (CPU模式)...")
    latent_vec = get_latent_features(user_prompt)
    feature_status = "✅ 已提取" if latent_vec is not None else "⚠️ 未載入"

    # === 步驟 3: 精準文本檢索 (TF-IDF Fallback) ===
    yield f'<p style="color: #6c757d;">🔎 正在全量檢索資料庫正文 (此步驟耗時較長，請稍候)...</p>'
    try:
        STATS_CSV = os.path.join(config.DATASET_DIR, "legal_stats.csv")
        target_csv = STATS_CSV if os.path.exists(STATS_CSV) else config.CSV_DATASET_PATH
        
        if os.path.exists(target_csv):
            # 已將 pandas 等遷移至頂部以加速執行
            df_rag = pd.read_csv(target_csv)
            
            # --- [效能革命：兩段式快篩架構] ---
            # 階段 2-A: 關鍵字快篩 (Candidate Pruning)
            # 針對文件進行字串掃描
            keywords = [k for k in ai_search_keywords.split() if len(k) > 1]
            if keywords:
                # 建立篩選遮罩
                mask = df_rag['TextContent'].str.contains(keywords[0], na=False, case=False)
                for k in keywords[1:5]: # 最多取 5 個關鍵字聯集
                    mask |= df_rag['TextContent'].str.contains(k, na=False, case=False)
                
                # 取得候選子集 (正式模式：移除筆數限制，確保全庫擴充後的完整召回)
                df_candidates = df_rag[mask]
                if len(df_candidates) == 0: 
                    df_candidates = df_rag
            else:
                df_candidates = df_rag # 無關鍵字則使用全庫進行檢索

            all_texts = df_candidates['TextContent'].fillna("").tolist()
            filenames = df_candidates['FileName'].tolist()
            
            # 使用包含 AI 關鍵字的 search_query 進行分詞與搜尋
            def tokenize_chinese(text):
                return " ".join(list(str(text)))
            
            all_texts_tokenized = [tokenize_chinese(t) for t in all_texts]
            # [核心修復] 僅使用 AI 提取的專業關鍵字進行 TF-IDF 檢索，撇除使用者指令雜訊
            query_tokenized = tokenize_chinese(ai_search_keywords)
            
            # --- [全字元極致模式] 字元長度擴充為 (1, 8) ---
            vectorizer = TfidfVectorizer(
                max_features=40000, 
                token_pattern=r"(?u)\b\w+\b", 
                ngram_range=(1, 8), 
                min_df=1, 
                max_df=0.9
            )
            
            # 啟動步驟 2
            yield f'<!-- PROGRESS: 正在生成深度文字矩陣 (N-gram 1-8)... -->'
            tfidf_matrix = vectorizer.fit_transform(all_texts_tokenized)
            query_vec = vectorizer.transform([query_tokenized])
            
            # 更新步驟 2 為完成，啟動步驟 3 (將腳本合併為單一字串，防止 chunk 斷裂導致 UI 顯示 raw JS)
            yield f'<script>document.getElementById("step-2").innerHTML = "✅ 階段 2: 深度文字矩陣生成完成 (N-gram 1-8)"; document.getElementById("step-2").className = "step-done"; document.getElementById("step-3").className = "step-active pulse";</script>'
            yield padding

            yield f'<!-- PROGRESS: 正在計算與 {len(all_texts)} 份文件的相似度... -->'
            cosine_sim = cosine_similarity(query_vec, tfidf_matrix).flatten()
            
            # 取得進行召回 (Recall) - 改為「全量比對」，不限筆數
            relevant_candidates = []
            
            # [召回優化] 移除強制 Top-5 補位，採用「實力說話」門檻 (0.05)
            # 因為 N-gram (1, 8) 分數較低，0.05 左右即代表有高度關鍵字重疊
            threshold = 0.05
            passed_indices = np.where(cosine_sim > threshold)[0]
            
            # 更新步驟 3 為完成，啟動步驟 4 (合併腳本防止分片導致 UI 顯示 raw JS)
            yield f'<script>document.getElementById("step-3").innerHTML = "✅ 階段 3: 全庫掃描完成，尋獲 {len(passed_indices)} 份相關文件"; document.getElementById("step-3").className = "step-done"; document.getElementById("step-4").className = "step-active pulse";</script>'
            yield f'<!-- PROGRESS: 找到 {len(passed_indices)} 份潛在資料，準備語意排序... -->'
            yield padding
            
            # 依分數從高到低排序這些索引
            sorted_passed_indices = passed_indices[np.argsort(cosine_sim[passed_indices])[::-1]]
            
            for idx in sorted_passed_indices:
                score = float(cosine_sim[idx])
                content = str(all_texts[idx])
                fname = filenames[idx]
                db_crime_type = str(df_candidates.iloc[idx].get('CrimeType', '未知'))
                
                relevant_candidates.append({
                    "fname": fname,
                    "content": content,  # 改為無上限完整內文
                    "type": db_crime_type,
                    "tfidf_score": score
                })

            # === 步驟 2.2: 使用您的【地端訓練模型】進行語意重排 (Semantic Reranking) ===
            relevant_docs = []
            if relevant_candidates:
                # 取得問題的地端模型特徵
                q_feat = get_latent_features(user_prompt)
                
                if q_feat is not None:
                    # 計算候選文件的地端模型特徵並比對
                    total_cand = len(relevant_candidates)
                    for i, doc in enumerate(relevant_candidates):
                        # 每 20 筆回傳一次進度，避免使用者以為當機
                        if (i+1) % 20 == 0:
                            msg = f"正在進行深度語意比對 ({i+1}/{total_cand})"
                            yield f'<!-- PROGRESS: {msg} -->'
                            yield f'<script>document.getElementById("step-4").innerHTML = "● 階段 4: {msg}...";</script>'
                            yield padding
                        
                        d_feat = get_latent_features(doc['content'])
                        if d_feat is not None:
                            # --- [Google TurboQuant 向量比對模式] ---
                            if tq_engine:
                                # 使用 4-bit TurboQuant 進行亞量化相似度估算 (模擬量化搜尋)
                                # 在此情境下，我們使用 TQ 的 inner_product 核心來模擬地端高效比對
                                q_tensor = torch.from_numpy(np.array(q_feat)).float()
                                d_tensor = torch.from_numpy(np.array(d_feat)).float()
                                
                                # 對候選向量進行 TQ 壓縮與估算
                                compressed_d = tq_engine.quantize(d_tensor)
                                sim = float(tq_engine.inner_product(q_tensor, compressed_d))
                            else:
                                # 原生 FP32 餘弦相似度
                                sim = float(cosine_similarity([q_feat], [d_feat])[0][0])
                            
                            doc['semantic_score'] = sim
                        else:
                            doc['semantic_score'] = 0.0
                    
                    relevant_candidates = sorted(relevant_candidates, key=lambda x: x['semantic_score'], reverse=True)
                    yield f'<script>document.getElementById("step-4").innerHTML = "✅ 階段 4: 語意重排完成，已精選最佳參考判項"; document.getElementById("step-4").className = "step-done"; document.getElementById("step-5").className = "step-active pulse";</script>'
                    yield padding
                else:
                    # 若模型未載入，則退而求其次使用 TF-IDF 排序
                    relevant_candidates = sorted(relevant_candidates, key=lambda x: x['tfidf_score'], reverse=True)
                    yield f'<script>'
                    yield f'document.getElementById("step-4").innerHTML = "⚠️ 階段 4: 未偵測到本地模型，改用 TF-IDF 全量排序完成";'
                    yield f'document.getElementById("step-4").className = "step-done";'
                    yield f'document.getElementById("step-5").className = "step-active pulse";'
                    yield f'</script>'
                    yield padding

                # --- [修正] 依照「字數上限」動態選取最強內容 ---
                current_chars = 0
                seen_files = set()
                
                for doc in relevant_candidates:
                    # 僅檢查總字數上限 (由 hardware/model 限制)，不再限制件數
                    if current_chars > config.RAG_MAX_CONTEXT_CHARS: break
                    
                    # 過濾重複檔名
                    base_name = doc['fname']
                    if base_name in seen_files: continue
                    seen_files.add(base_name)
                    
                    relevant_docs.append(doc)
                    current_chars += len(doc['content'])

            if relevant_docs:
                source_documents_html = "<ul>"
                for i, doc in enumerate(relevant_docs):
                    rag_context += f"【參考資料 {i+1} - {doc['fname']} (庫存案由: {doc['type']})】:\n{doc['content']}\n\n"
                    source_documents_html += f"<li><strong>{doc['fname']}</strong></li>"
                source_documents_html += "</ul>"
                
                # --- [除錯心法] 在控制台印出 RAG Context 預覽 ---
                print(f"\n--- [DEBUG] RAG Context Preview (Total {len(relevant_docs)} docs) ---")
                print(rag_context[:1000] + "..." if len(rag_context) > 1000 else rag_context)
                print("---------------------------------------------------\n")
            else:
                source_documents_html = "<p style='color: #666;'>根據「案由精準過濾」，目前資料庫中無與該罪名直接相符的判例。</p>"
                
    except Exception as e:
        source_documents_html = f"<span style='color: red;'>檢索異常: {e}</span>"

    # === 步驟 3: 組合精簡強效 Prompt 給 Ollama (Llama) ===
    try:
        # 針對 Llama 3.1 8B 進行「高對等性」與「反幻覺」優化
        system_prompt = f"""
您是台灣法律諮詢專家。請針對使用者的問題：「{user_prompt}」，結合以下資料提供分析。

【⚠ 重要反幻覺與指令遵循準則】：
1. **指令優先**：使用者問題中若包含「請告知我是參照哪一個檔案」、「請列出具體案號」等指令，請**務必遵守**。
2. **主題相關性過濾 (核心防死鎖)**：請先核對【資料來源 1】的主體內容與使用者的「事實」是否具備實質相關性。
   - **如果完全不對題**（例如問性侵卻給光碟案），請明確告知：「目前地端資料庫查無與此特定情節相關的判例，暫無法基於資料庫內容進行分析。」，**絕對禁止** 強行套用不對題的案例進行法律論證。
3. **事實對位分析**：如果資料相關，請拿使用者描述的事實與【資料來源 1】中的情節進行比對分析。

【資料來源 1：地端 PDF 判例摘要 (優先參考)】：
{rag_context if rag_context else "（目前地端資料庫查無高度相符的案例片斷）"}

【資料來源 2：台灣法律體系發散背景】：
- 請將上述案例與您的法律知識（民事、刑事、家事、性侵防治法等）對齊，給予具體的定罪建議或行動方針。

【回覆準則】：
- **必須明確指出檔名**（如：XXX.pdf）以便使用者核對。
- **格式規範 (HTML ONLY)**：**絕對禁止** 使用 Markdown。
  - 標題用 <p><strong>文字</strong></p>，內文用 <p>文字</p>。
"""

        # 先 Yield 報告的開頭與來源部分
        header_html = f'''
        <div style="font-family: 'Segoe UI', Arial, sans-serif; line-height: 1.6; color: #333;">
            <h2 style="color: #2c3e50; text-align: center; border-bottom: 3px solid #34495e; padding-bottom: 10px;">⚖️ 智慧法律深度分析報告 (RAG + Stats)</h2>
            
            {stats_html}

            <div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 2px solid #ddd; padding-bottom: 5px;">
                <h3 style="color: #444; margin: 0;">📚 知識庫檢索來源</h3>
                <span style="font-size: 0.8em; color: #666;">地端特徵提取: {feature_status}</span>
            </div>
            <div style="padding: 10px 20px; border-radius: 8px; background-color: #e9ecef; border-left: 5px solid #6c757d; font-size: 0.9em; margin-bottom: 20px; margin-top: 10px;">
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
        
        # 增加超時至 1200 秒 (20 分鐘)，確保效能較弱的電腦也能跑完分析
        with urllib.request.urlopen(req, timeout=1200) as response:
            import re
            for line in response:
                if line:
                    chunk = json.loads(line.decode('utf-8'))
                    response_text = chunk.get("response", "")
                    if response_text:
                        # --- [修正] 改用針對性濾網，避免移除 HTML 標籤的 > ---
                        cleaned_chunk = response_text.replace(">>", "").replace("###", "").replace("**", "").replace("__", "")
                        # 處理行首引用符號，加個空格判斷比較保險
                        cleaned_chunk = cleaned_chunk.replace("\n>", "\n").replace(" > ", " ")
                        yield cleaned_chunk
                        if chunk.get("done", False):
                            break
        
        # 標記最後階段完成
        yield f'<script>document.getElementById("step-5").innerHTML = "✅ 階段 5: Llama 總結分析與建議產出完成"; document.getElementById("step-5").className = "step-done";</script>'
        yield padding

        # 結尾 HTML
        footer_html = f'''
            </div>
            <div style="margin-top: 40px; padding: 10px; background-color: #e2e3e5; border-radius: 5px; text-align: center;">
                <p style="color: #6a6c6f; font-size: 0.85em; margin-bottom: 0;">
                    本次分析由 <b>Full Corpus RAG (ChromaDB + llama)</b> 驅動。所有內容皆在本地生成。
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
