import os
import sys
import glob
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import torch
from langchain_core.embeddings import Embeddings
import torch
from langchain_core.embeddings import Embeddings
import config
from models import Vocab, LegalAutoencoder

class LocalLegalEmbedding(Embeddings):
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_path = os.path.join(config.BASE_DIR, "dataset", "best_model.pth")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"找不到模型檔案: {model_path}，請先執行訓練。")
        
        checkpoint = torch.load(model_path, map_location=self.device)
        # 使用新的 Vocab 初始化方式
        self.vocab = Vocab(checkpoint['vocab'], checkpoint['inv_vocab'])
        self.model = LegalAutoencoder(len(self.vocab.vocab))
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()

    def embed_documents(self, texts):
        embeddings = []
        for text in texts:
            embeddings.append(self.embed_query(text))
        return embeddings

    def embed_query(self, text):
        indices = self.vocab.encode(text, config.TRAIN_MAX_SEQ_LEN)
        input_tensor = torch.tensor([indices], dtype=torch.long).to(self.device)
        with torch.no_grad():
            _, latent = self.model(input_tensor)
        return latent.cpu().numpy()[0].tolist()

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
    print("\n正在載入『地端專屬法律特徵模型』(best_model.pth)...")
    
    print("\n正在載入『地端專屬法律特徵模型』作為 Embedding 引擎...")
    embeddings = LocalLegalEmbedding()
    
    print("\n正在建立本地端 ChromaDB 向量知識庫 (使用專屬特徵)...")
    from langchain_community.vectorstores import Chroma
    
    # 清除舊的資料庫以確保索引空間一致 (不一致會導致搜尋異常)
    if os.path.exists(db_dir):
        import shutil
        shutil.rmtree(db_dir)
        print(f"已清理舊有資料庫以進行全文重新索引。")

    vectorstore = Chroma.from_documents(
        documents=splits, 
        embedding=embeddings, 
        persist_directory=db_dir
    )
    
    print(f"\n✅ 知識庫建置完成！已成功儲存至大腦記憶區：{db_dir}")
    print("👉 系統現在具備了針對全部判決書內容的檢索與回答能力！")

if __name__ == "__main__":
    build_vector_database()
