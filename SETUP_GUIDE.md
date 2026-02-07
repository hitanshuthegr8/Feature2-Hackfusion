# JournalSense - Setup and Run Guide

This guide will help you set up and run the JournalSense AI Research Assistant Platform.

## 📋 Prerequisites

Before you begin, ensure you have the following installed:

- **Node.js** (v16 or higher) - [Download here](https://nodejs.org/)
- **Python** (v3.8 or higher) - [Download here](https://www.python.org/downloads/)
- **npm** or **yarn** (comes with Node.js)
- **pip** (comes with Python)

## 🚀 Step-by-Step Setup Instructions

### Part 1: Backend Setup (Python/Streamlit)

The backend consists of Streamlit applications for journal recommendations, dashboard, and keyword finding.

#### Step 1: Navigate to the Models directory
```bash
cd Models
```

#### Step 2: Create a virtual environment (Recommended)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

#### Step 3: Install Python dependencies
```bash
pip install -r requirements.txt
```

**Note:** The requirements.txt includes a spaCy model. If installation fails, you may need to install it separately:
```bash
python -m spacy download en_core_web_sm
```

Alternatively, if the direct download doesn't work:
```bash
pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.0/en_core_web_sm-2.2.0.tar.gz
```

#### Step 4: Run the Streamlit application

You have three Streamlit apps to choose from:

**Main Journal Recommender (Recommended):**
```bash
streamlit run model2.py
```

**Dashboard:**
```bash
streamlit run dashboard.py
```

**Keyword Finder:**
```bash
streamlit run keywordFinder.py
```

The application will start on `http://localhost:8501` by default.

---

### Part 2: Frontend Setup (React/TypeScript)

The project has two frontend applications:
1. Root directory frontend (landing page)
2. `project/` directory frontend (main application)

#### Option A: Run Root Directory Frontend

#### Step 1: Navigate to project root
```bash
cd ..  # If you're in Models directory
# or navigate to the project root directory
```

#### Step 2: Install dependencies
```bash
npm install
```

#### Step 3: Start the development server
```bash
npm run dev
```

The application will start on `http://localhost:5173` (default Vite port).

---

#### Option B: Run Project Subdirectory Frontend

#### Step 1: Navigate to project subdirectory
```bash
cd project
```

#### Step 2: Install dependencies
```bash
npm install
```

#### Step 3: Start the development server
```bash
npm run dev
```

The application will start on `http://localhost:5173` (default Vite port).

---

## 🔧 Configuration Notes

### API Keys

The frontend uses a Gemini API key (currently hardcoded in `project/src/services/gemini.ts`). For production use, consider:
- Moving the API key to environment variables
- Using a `.env` file with `VITE_API_KEY=your_key_here`
- Updating the service to read from environment variables

### Port Configuration

- **Streamlit apps**: Default port `8501`
- **Vite frontend**: Default port `5173`

If ports are in use, Streamlit will automatically use the next available port, and Vite will prompt you to use a different port.

---

## 📝 Available Scripts

### Frontend (npm scripts)
- `npm run dev` - Start development server
- `npm run build` - Build for production
- `npm run preview` - Preview production build
- `npm run lint` - Run ESLint

### Backend (Streamlit)
- `streamlit run <filename>.py` - Run a Streamlit app
- `streamlit run <filename>.py --server.port 8502` - Run on custom port

---

## 🐛 Troubleshooting

### Python/Streamlit Issues

1. **spaCy model not found:**
   ```bash
   python -m spacy download en_core_web_sm
   ```

2. **FAISS installation fails:**
   - On Windows, you might need: `pip install faiss-cpu`
   - On macOS/Linux: `pip install faiss-cpu` or `pip install faiss` (if you have CUDA)

3. **Module not found errors:**
   - Ensure virtual environment is activated
   - Reinstall requirements: `pip install -r requirements.txt --upgrade`

### Frontend Issues

1. **Port already in use:**
   - Kill the process using the port, or
   - Use a different port: `npm run dev -- --port 3000`

2. **Dependencies not installing:**
   - Clear cache: `npm cache clean --force`
   - Delete `node_modules` and `package-lock.json`, then reinstall

3. **TypeScript errors:**
   - Run `npm run lint` to see specific issues
   - Ensure all dependencies are installed

---

## 🌐 Running Both Frontend and Backend

To run the complete application:

1. **Terminal 1 - Backend:**
   ```bash
   cd Models
   # Activate virtual environment if using one
   streamlit run model2.py
   ```

2. **Terminal 2 - Frontend:**
   ```bash
   cd project  # or root directory
   npm run dev
   ```

Both will run simultaneously, and you can access:
- Frontend: `http://localhost:5173`
- Backend: `http://localhost:8501`

---

## 📚 Project Structure

```
JournalSense--AI-Research-assistant-platform/
├── Models/                    # Python backend (Streamlit apps)
│   ├── model2.py             # Main journal recommender
│   ├── dashboard.py           # Dashboard app
│   ├── keywordFinder.py       # Keyword finder app
│   └── requirements.txt       # Python dependencies
├── project/                   # React frontend (main app)
│   ├── src/
│   │   ├── components/
│   │   ├── services/
│   │   └── types/
│   └── package.json
├── src/                       # React frontend (landing page)
└── package.json               # Root frontend dependencies
```

---

## ✅ Quick Start (TL;DR)

**Backend:**
```bash
cd Models
pip install -r requirements.txt
python -m spacy download en_core_web_sm
streamlit run model2.py
```

**Frontend:**
```bash
cd project
npm install
npm run dev
```

---

## 🆘 Need Help?

If you encounter issues:
1. Check that all prerequisites are installed correctly
2. Ensure you're using the correct Python/Node.js versions
3. Verify all dependencies are installed
4. Check the console/terminal for specific error messages

Happy coding! 🚀

