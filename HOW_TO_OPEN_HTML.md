# How to Open HTML File (CORS Fix)

## ❌ DON'T Do This:
- Double-clicking `test_ui.html` 
- Opening via `file:///C:/Users/.../test_ui.html`
- This causes CORS errors!

## ✅ DO This Instead:

### Method 1: Python HTTP Server (Easiest)

1. **Open terminal in project directory:**
   ```bash
   cd C:\Users\HP\Desktop\Secret\Project\ResearchAss
   ```

2. **Start web server:**
   ```bash
   python -m http.server 8000
   ```

3. **Open in browser:**
   ```
   http://localhost:8000/test_ui.html
   ```

4. **Keep terminal open** - server runs until you press `Ctrl+C`

---

### Method 2: VS Code Live Server (Recommended)

1. **Install Live Server extension** in VS Code
2. **Right-click** on `test_ui.html`
3. **Select**: "Open with Live Server"
4. **Browser opens automatically** at `http://127.0.0.1:5500/test_ui.html`

---

### Method 3: Node.js http-server

If you have Node.js installed:

```bash
# Install globally (one time)
npm install -g http-server

# Run in project directory
http-server -p 8000

# Open: http://localhost:8000/test_ui.html
```

---

### Method 4: Python Flask (Alternative)

Create a simple Flask server:

```python
# serve_html.py
from flask import Flask, send_from_directory

app = Flask(__name__)

@app.route('/')
def index():
    return send_from_directory('.', 'test_ui.html')

if __name__ == '__main__':
    app.run(port=8000, debug=True)
```

Run: `python serve_html.py`

---

## Quick Start (Recommended)

**Just run this command:**
```bash
python -m http.server 8000
```

**Then open:** `http://localhost:8000/test_ui.html`

That's it! ✅

---

## Verify It's Working

1. **Check browser address bar:**
   - ✅ Should show: `http://localhost:8000/test_ui.html`
   - ❌ NOT: `file:///C:/Users/...`

2. **Check browser console (F12):**
   - ✅ Should show: `✅ Server is running`
   - ❌ NOT: CORS errors

3. **Check status indicator:**
   - ✅ Should show: `Server: Online`
   - ❌ NOT: `Server: Offline`

---

## Why This Matters

- `file://` protocol blocks CORS requests for security
- Web server (`http://`) allows CORS to work properly
- Your Flask backend expects requests from `http://` origin

---

## Troubleshooting

**Port 8000 already in use?**
```bash
# Use different port
python -m http.server 8080
# Then open: http://localhost:8080/test_ui.html
```

**Python not found?**
- Make sure Python is installed and in PATH
- Try: `python3 -m http.server 8000` (Mac/Linux)
- Or: `py -m http.server 8000` (Windows)

