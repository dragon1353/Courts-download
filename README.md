# ⚖️ Taiwan Judicial Judgment PDF Downloader (司法院判決書自動下載器)

這是一個基於 Python 與 Playwright 開發的桌面應用程式，旨在幫助法律從業人員或資料科學家快速從「司法院法學資料檢索系統」自動化下載指定日期區間的判決書 PDF 檔案。

## ✨ 特色功能
- **直觀 GUI 介面**：使用桌面視窗操作，無需接觸複雜的程式碼指令。
- **自動化跨頁爬取**：內建智慧分頁邏輯，自動處理「下一頁」跳轉，突破單次檢索上限。
- **自訂日期區間**：完全支援民國年格式輸入，精準鎖定下載範圍。
- **地端隱私保護**：所有下載流程皆在本機端執行，不經過任何第三方雲端服務。

## 🛠️ 技術堆棧
- **Frontend**: HTML5, Vanilla CSS, JS (純電原生設計)
- **Backend**: Flask (輕量級 Web 服務)
- **Engine**: Playwright (高效能瀏覽器自動化)
- **UI Framework**: PyWebView (Native 桌面視窗封裝)
  
📝 聲明
本工具僅供學術研究與法律資料分析使用。使用時請遵守司法院網站之相關使用規範與爬蟲禮儀。

```bash
# 安裝必要套件
pip install flask pywebview playwright pypdf
playwright install chromium

Roadmap: Local AI Model Training Integration (Coming Soon)
