# How to Run the Backend Server

## Quick Start (If dependencies are already installed)

```bash
python run_server.py
```

The server will start on `http://localhost:5000`

---

## Full Setup (First Time)

### Step 1: Activate Virtual Environment

**Windows:**
```bash
.\venv\Scripts\activate
```

**Mac/Linux:**
```bash
source venv/bin/activate
```

### Step 2: Install Dependencies (if not already installed)

```bash
pip install -r backend/requirements.txt
```

### Step 3: Download spaCy Model (if not already installed)

```bash
python -m spacy download en_core_web_sm
```

**Note:** The `run_server.py` script will automatically try to install this if missing.

### Step 4: Run the Server

```bash
python run_server.py
```

---

## What You'll See

When the server starts successfully, you'll see:

```
============================================================
[*] JournalSense API Server
============================================================

Endpoints:
  POST /upload-pdf    - Upload and process PDF
  POST /search-topic  - Search by research topic
  GET  /papers        - Get all indexed papers
  GET  /papers/<id>   - Get specific paper
  GET  /compare       - Compare all papers (gap analysis)
  GET  /explain/<id>/<entity> - Get cursor trace
  POST /search        - Semantic search

============================================================

 * Running on http://0.0.0.0:5000
 * Debug mode: on
```

---

## Verify Server is Running

Open your browser and go to:
- Health check: `http://localhost:5000/health`

Or test with curl:
```bash
curl http://localhost:5000/health
```

---

## Troubleshooting

### Port Already in Use
If port 5000 is already in use, you can change it in `run_server.py`:
```python
app.run(host='0.0.0.0', port=5001, debug=True)  # Change port number
```

### Missing Dependencies
If you get import errors, make sure all dependencies are installed:
```bash
pip install -r backend/requirements.txt
```

### spaCy Model Error
If spaCy model is missing:
```bash
python -m spacy download en_core_web_sm
```

### Virtual Environment Not Activated
Make sure you see `(venv)` in your terminal prompt before running the server.

---

## Stop the Server

Press `Ctrl+C` in the terminal where the server is running.

