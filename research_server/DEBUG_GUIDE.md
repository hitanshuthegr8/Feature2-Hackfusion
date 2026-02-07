# 🚀 Research Paper Analyzer - Testing & Debug Guide

## Quick Start Commands

### Option 1: Automated Start (Recommended)
```batch
# Just double-click this file:
start_servers.bat
```

### Option 2: Manual Start
```powershell
# Terminal 1 - Backend API
cd c:\Users\HP\Desktop\Secret\Project\Compare_RP\research_server
.\venv\Scripts\python.exe app.py

# Terminal 2 - Frontend Server
cd c:\Users\HP\Desktop\Secret\Project\Compare_RP\research_server\simple_client
python -m http.server 8080
```

---

## 🔍 Debug Commands

### 1. Test Backend Health
```powershell
Invoke-WebRequest -Uri "http://localhost:5000/health" -UseBasicParsing | Select-Object -ExpandProperty Content | ConvertFrom-Json
```

### 2. Check All Services Status
```powershell
Invoke-WebRequest -Uri "http://localhost:5000/pipeline/status" -UseBasicParsing | Select-Object -ExpandProperty Content | ConvertFrom-Json | ConvertTo-Json -Depth 3
```

### 3. Test Search Only (Fast, No Processing)
```powershell
$body = '{"query": "transformer vision"}';
Invoke-WebRequest -Uri "http://localhost:5000/search" -Method POST -ContentType "application/json" -Body $body -UseBasicParsing | Select-Object -ExpandProperty Content | ConvertFrom-Json | ConvertTo-Json -Depth 3
```

### 4. Full Pipeline Test (1 Paper)
```powershell
$body = '{"query": "deep learning", "max_papers": 1}';
Invoke-WebRequest -Uri "http://localhost:5000/pipeline/analyze" -Method POST -ContentType "application/json" -Body $body -UseBasicParsing -TimeoutSec 180 | Select-Object -ExpandProperty Content | Out-File -FilePath "pipeline_result.json" -Encoding UTF8;
Get-Content pipeline_result.json | ConvertFrom-Json | ConvertTo-Json -Depth 5
```

### 5. View Backend Logs (Real-time)
```powershell
# The backend will show detailed [DEBUG] logs in the terminal
# Look for lines starting with:
# - [DEBUG] Starting extraction...
# - [DEBUG] Ollama response received...
# - [DEBUG] ✅ Extraction complete
# - [DEBUG] ❌ Failed to parse...
```

---

## 📊 What to Look For in Logs

### ✅ **SUCCESS Pattern**
```
[DEBUG] Starting extraction on text of length: XXXX chars
[DEBUG] Prompt length: YYYY chars
[DEBUG] Calling Ollama generate...
[DEBUG] ✅ Ollama response received: ZZZZ chars
[DEBUG] Response preview (first 500 chars):
{
  "models": ["ResNet-50", "U-Net"],
  "datasets": ["ImageNet", "CIFAR-10"],
  "baselines": [...]
}
[DEBUG] Parsing JSON...
[DEBUG] ✅ JSON parsed successfully: <class 'dict'>
[DEBUG] Extracted keys: ['models', 'datasets', 'baselines']
[DEBUG] ✅ Extraction complete:
[DEBUG]   - Models: 2 found -> ['ResNet-50', 'U-Net']
[DEBUG]   - Datasets: 2 found -> ['ImageNet', 'CIFAR-10']
[DEBUG]   - Baselines: 1 found -> [{'metric': 'Accuracy', 'value': '95.2%'}]
```

### ❌ **FAILURE Patterns to Report**

#### Pattern 1: Empty Ollama Response
```
[DEBUG] Calling Ollama generate...
[DEBUG] ❌ Ollama returned empty response!
```
**Action**: Check if Ollama is running (`ollama list`)

#### Pattern 2: JSON Parse Error
```
[DEBUG] ✅ Ollama response received: 234 chars
[DEBUG] Response preview (first 500 chars):
I found the following models: ...  <-- NOT JSON!
[DEBUG] ❌ Failed to parse Ollama JSON response: ...
[DEBUG] Full raw response:
<FULL TEXT HERE>
```
**Action**: Share the full response - Ollama might not be following JSON format

#### Pattern 3: Wrong Data Type
```
[DEBUG] ✅ JSON parsed successfully: <class 'str'>  <-- Should be dict!
[DEBUG] ❌ Ollama response is not a dictionary, got: <class 'str'>
```
**Action**: Ollama is returning JSON string instead of object

#### Pattern 4: Empty Lists
```
[DEBUG] ✅ Extraction complete:
[DEBUG]   - Models: 0 found -> []
[DEBUG]   - Datasets: 0 found -> []
[DEBUG]   - Baselines: 0 found -> []
```
**Action**: Ollama is returning valid JSON but empty arrays - might need better prompt

---

## 🐛 Common Issues & Fixes

### Issue 1: "0 Models, 0 Datasets" (Your Current Issue)
**Symptoms**: Papers processed but extraction returns empty
**Debug Steps**:
1. Run pipeline with 1 paper
2. Check backend terminal for `[DEBUG]` logs
3. Look for the "Response preview" - is it valid JSON?
4. Share the full log output here

### Issue 2: PDF Download Failures
**Symptoms**: `Failed to download PDF` errors
**Cause**: Paywalls, anti-bot protection (418 errors)
**Fix**: Try queries with open-access papers:
- "arXiv machine learning"
- "bioRxiv computational biology"
- "PubMed Central medical imaging"

### Issue 3: GROBID 500 Errors
**Symptoms**: `GROBID API request failed: 500`
**Cause**: Cloud service overloaded or PDF invalid
**Fix**: Retry later or use different papers

### Issue 4: Ollama Offline
**Symptoms**: `Ollama: offline` in status
**Fix**:
```powershell
# Check if Ollama is running
ollama list

# Start Ollama if needed
ollama serve

# Pull models if missing
ollama pull llama3.1:8b
```

---

## 📝 How to Send Debug Logs

### Quick Log Capture
```powershell
# Run test and save all output
$body = '{"query": "transformer", "max_papers": 1}';
Invoke-WebRequest -Uri "http://localhost:5000/pipeline/analyze" -Method POST -ContentType "application/json" -Body $body -UseBasicParsing -TimeoutSec 180 2>&1 | Out-File -FilePath "debug.log" -Encoding UTF8

# Copy backend terminal output to a file manually
# Then share debug.log + backend terminal output
```

---

## 🎯 Expected Behavior

For query: `"deep learning image classification"` with 2 papers:

1. **TF-IDF** extracts: `["deep", "learning", "image", "classification"]`
2. **OpenAlex** finds: 25 papers, selects 2 with PDFs
3. **PDF Download**: Attempts download (may fail for paywalled)
4. **GROBID**: Parses PDF → JSON (may fail if PDF invalid)
5. **Ollama**: Extracts from JSON text
   - Should return: `{"models": [...], "datasets": [...], "baselines": [...]}`
6. **Comparison**: Aggregates all results

**Current Issue**: Step 5 (Ollama) returns empty arrays `[]`

---

## 🚀 Next Steps

1. **Start servers** using `start_servers.bat`
2. **Open browser**: http://localhost:8080
3. **Run test query** with 1-2 papers
4. **Check backend terminal** for `[DEBUG]` logs
5. **Copy all logs** starting from "Starting extraction..." to "Extraction complete"
6. **Share logs** so I can diagnose the exact issue

The detailed debug logs will show exactly where the extraction is failing!
