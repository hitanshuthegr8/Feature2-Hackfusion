# Troubleshooting: "Cannot connect to server" Error

## Common Causes and Solutions

### 1. HTML Opened as file:// (Most Common)

**Problem**: If you double-click `test_ui.html`, it opens as `file://` which causes CORS to block all API requests.

**Solution**: Use a web server instead:

```bash
# In the project directory
python -m http.server 8000
```

Then open: `http://localhost:8000/test_ui.html`

**Or use VS Code Live Server extension**

---

### 2. Server Not Running

**Check if server is running:**
```bash
# Windows
netstat -ano | findstr :5000

# Should show something like:
# TCP    0.0.0.0:5000           0.0.0.0:0              LISTENING
```

**Start the server:**
```bash
python run_server.py
```

---

### 3. Port Already in Use

**Check what's using port 5000:**
```bash
# Windows
netstat -ano | findstr :5000
```

**Kill the process** (replace PID with the number from above):
```bash
taskkill /PID <PID> /F
```

**Or change the port** in `run_server.py`:
```python
app.run(host='0.0.0.0', port=5001, debug=True)  # Change to 5001
```

And update `test_ui.html`:
```javascript
const API = 'http://localhost:5001';  // Match the port
```

---

### 4. CORS Issues

**Check browser console (F12)** for CORS errors.

**Verify CORS is enabled** in `backend/app.py`:
```python
CORS(app, resources={r"/*": {"origins": "*"}})
```

If you see CORS errors, make sure:
- HTML is opened via web server (not file://)
- Server is running
- CORS is enabled in backend

---

### 5. Firewall/Antivirus Blocking

**Temporarily disable** firewall/antivirus to test, or:
- Allow Python through firewall
- Allow port 5000 through firewall

---

### 6. Browser Cache

**Clear browser cache** or use **Incognito/Private mode**

---

## Quick Diagnostic Steps

1. **Open browser console (F12)** - Check for detailed error messages
2. **Check server logs** - Look for errors in the terminal where server is running
3. **Test health endpoint directly**:
   ```bash
   curl http://localhost:5000/health
   ```
   Or open in browser: `http://localhost:5000/health`

4. **Check the HTML file** - Make sure it's opened via web server, not file://

---

## Expected Behavior

When everything works:
- Browser console shows: `✅ Server is running`
- Status indicator shows: `Server: Online`
- Health check returns JSON with server status

---

## Still Not Working?

1. Check browser console (F12) for exact error
2. Check server terminal for errors
3. Verify server is accessible: `http://localhost:5000/health`
4. Make sure HTML is opened via web server (not file://)

