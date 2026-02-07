# ✅ Server Configuration Ready

## Current Status

- **Flask Server**: Running on port 5000 (PID: 31908)
- **HTML File**: `test_ui.html` is configured correctly
- **API Endpoint**: `http://localhost:5000`

## HTML Configuration ✅

The `test_ui.html` file is correctly set up:

```javascript
const API = 'http://localhost:5000';
```

All endpoints match the Flask backend:
- ✅ `POST /upload-pdf` - Upload PDF
- ✅ `POST /build-corpus` - Build corpus
- ✅ `POST /rag/analyze` - RAG analysis
- ✅ `POST /rag/gaps` - Gap analysis
- ✅ `POST /grounded-gaps` - Grounded gaps (no LLM)
- ✅ `POST /rag/ask` - Q&A
- ✅ `POST /rag/review` - Peer review
- ✅ `GET /health` - Health check
- ✅ `GET /papers` - Get papers

## How to Use

1. **Make sure Flask server is running**:
   ```bash
   python run_server.py
   ```

2. **Open the HTML file**:
   - Option A: Use a local web server (recommended):
     ```bash
     python -m http.server 8000
     ```
     Then open: `http://localhost:8000/test_ui.html`
   
   - Option B: Use VS Code Live Server extension
   
   - Option C: Open directly (may have CORS issues):
     Double-click `test_ui.html`

3. **Test the connection**:
   - The page will automatically check server health on load
   - Check browser console (F12) for connection status
   - Status indicators at top show server/ollama status

## Troubleshooting

If you get "Failed to fetch":
1. ✅ Check server is running: `netstat -ano | findstr :5000`
2. ✅ Open HTML via web server (not file://)
3. ✅ Check browser console (F12) for detailed errors
4. ✅ Verify CORS is enabled (it is: `CORS(app, resources={r"/*": {"origins": "*"}})`)

## All Set! 🚀

Your HTML is ready to connect to the Flask server. Just make sure:
- Flask server is running on port 5000
- Open HTML via web server (not file://)
- Check browser console for any errors

