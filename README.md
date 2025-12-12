# Face Enhancer Service

FastAPI 微服務，提供臉部增強 (Face Enhancement) 功能。模型會在首次使用時自動下載至 `.assets` 目錄。

## Quick start

1. 安裝依賴：
   ```bash
   cd face_enhancer
   pip install -r requirements.txt
   pip install -e .
   ```
2. 啟動 API：
   ```bash
   uvicorn face_enhancer.app:app --reload --port 8005
   ```
   開啟 `http://localhost:8005/` 使用 Demo UI。

3. 呼叫服務：
   ```bash
   curl -X POST http://localhost:8005/enhance \
     -H "Content-Type: application/json" \
     -d @payload.json  # 參考 /docs 取得 schema
   ```

## Docker

```bash
docker compose up --build
```

服務會在 `http://localhost:8005` 啟動，模型會持久化儲存。

## 預設值

- 執行裝置：GPU (CUDA)，若無則自動退回 CPU
- 模型：`gfpgan_1.4`
- Face Selector Mode：`reference`
