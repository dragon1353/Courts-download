import os
import sys
import glob
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# 確保載入 config
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config

def build_vector_database():
    pdf_dir = config.PDF_SAVE_PATH
    db_dir = os.path.join(config.BASE_DIR, "dataset", "chroma_db")
    
    pdf_files = glob.glob(os.path.join(pdf_dir, "*.pdf"))
    if not pdf_files:
        print(f"在 {pdf_dir} 找不到任何 PDF 檔案。請先下載判決書。")
        return

    print(f"找到 {len(pdf_files)} 個 PDF 檔案，正在載入並解析內容...")
    
    docs = []
    for i, pdf_path in enumerate(pdf_files):
        try:
            print(f"[{i+1}/{len(pdf_files)}] 處理: {os.path.basename(pdf_path)}...")
            loader = PyPDFLoader(pdf_path)
            docs.extend(loader.load())
        except Exception as e:
            print(f"讀取 {pdf_path} 失敗: {e}")

    print("\n正在切割文本 (Chunking)...")
    # 將長篇判決書切成小段落，以符合 LLM 大腦的吸收長度
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )
    splits = text_splitter.split_documents(docs)
    
    print(f"共切分為 {len(splits)} 個資料段落。")
    print("\n正在載入 Embedding 向量化模型 (初次執行需要下載)...")
    
    # 選擇適合繁體中文的輕量級開源模型 (intfloat/multilingual-e5-small)
    embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-small")
    
    print("\n正在建立本地端 ChromaDB 向量知識庫...")
    # 若資料庫已存在，這會直接寫入/覆蓋現有記憶
    vectorstore = Chroma.from_documents(
        documents=splits, 
        embedding=embeddings, 
        persist_directory=db_dir
    )
    
    print(f"\n✅ 知識庫建置完成！已成功儲存至大腦記憶區：{db_dir}")
    print("👉 系統現在具備了針對全部判決書內容的檢索與回答能力！")

if __name__ == "__main__":
    build_vector_database()
