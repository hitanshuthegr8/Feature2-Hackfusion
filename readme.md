# 🔬 JournalSense - AI Research Assistant Platform

> **Production-grade research intelligence pipeline with cursor-level explainability**

## 🎯 System Architecture

```
PDF / Topic
     ↓
Document Intelligence Layer (spaCy)
     ↓
Canonical Research JSON
     ↓
OpenAlex Expansion & Validation
     ↓
Vector Index (FAISS)
     ↓
Comparative Reasoning Engine
     ↓
Cursor-Explainable Outputs
```

## 🔑 Key Principle

> **Everything becomes JSON before anything becomes embeddings.**

## 🚀 Quick Start

### 1. Backend Setup

```bash
# Navigate to project root
cd ResearchAss

# Create virtual environment
python -m venv venv

# Activate (Windows)
.\venv\Scripts\activate

# Install dependencies
pip install -r backend/requirements.txt

# Download spaCy model
python -m spacy download en_core_web_sm

# Run server
python run_server.py
```

### 2. Frontend Setup

```bash
cd project
npm install
npm run dev
```

## 📡 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check with index stats |
| `/upload-pdf` | POST | Upload and process PDF → Canonical JSON |
| `/search-topic` | POST | Search OpenAlex by topic |
| `/papers` | GET | Get all indexed papers |
| `/papers/<id>` | GET | Get specific paper |
| `/compare` | GET | Comparative gap analysis |
| `/explain/<id>/<entity>` | GET | Cursor-level trace |
| `/search` | POST | Semantic search (FAISS) |
| `/clear` | POST | Clear index |

## 📊 Canonical Research JSON Schema

```json
{
  "paper_id": "local_001",
  "title": "...",
  "architecture": ["ViT", "UNet"],
  "modules": ["self-attention", "decoder"],
  "datasets": ["BraTS"],
  "metrics": {"Dice": 0.91},
  "baselines": ["UNet"],
  "tasks": ["segmentation"],
  "limitations": ["single dataset evaluation"],
  "intent_phrases": ["we propose", "to improve accuracy"],
  "raw_text_refs": {"method": "...", "results": "..."},
  "entity_traces": [...],
  "openalex": {
    "work_id": "...",
    "cited_by_count": 412,
    "publication_year": 2023,
    "concepts": ["Vision Transformer"],
    "trend_velocity": 137.3,
    "is_sota": true
  }
}
```

## 🏗️ Pipeline Phases

### Phase 1: Document Intelligence
- PDF → Text with section segmentation
- spaCy entity extraction (MODEL, DATASET, METRIC, TASK, BASELINE, LIMITATION)
- Keyword canonicalization (ViT → vision_transformer)

### Phase 2: OpenAlex Enrichment
- Concept expansion queries
- Citation metrics & trend velocity
- SoTA detection
- Benchmark coverage analysis

### Phase 3: FAISS Vectorization
- Embeds **structured summaries**, not raw text
- Section-aware indexing
- Cosine similarity search

### Phase 4: Comparative Analysis
- Architecture/dataset/baseline distributions
- Common patterns & evaluation gaps
- Novel opportunity suggestions

### Phase 5: Cursor Explainability
- Every insight → paper → section → character offset
- Click-to-navigate trace system

## 🎤 Judge Demo Script

> "We convert every paper into a canonical JSON before any reasoning.
> OpenAlex enriches it, FAISS retrieves it, and our comparison engine
> finds gaps with full cursor-level traceability."

## 📁 Project Structure

```
ResearchAss/
├── backend/
│   ├── __init__.py
│   ├── app.py              # Flask server
│   ├── config.py           # Settings & canonicalization maps
│   ├── pdf_extractor.py    # PDF → Sections
│   ├── entity_extractor.py # spaCy → Canonical JSON
│   ├── openalex_client.py  # OpenAlex integration
│   ├── faiss_index.py      # Vector indexing
│   ├── comparative_engine.py # Gap analysis
│   ├── explainability.py   # Cursor traces
│   └── requirements.txt
├── project/                 # React frontend
├── Models/                  # Streamlit apps (legacy)
├── run_server.py           # Server entry point
└── README.md
```

## ⚡ Tech Stack

| Component | Technology |
|-----------|------------|
| Backend | Flask + Python 3.10+ |
| PDF Extraction | PyMuPDF (fitz) |
| NLP | spaCy (en_core_web_sm) |
| Embeddings | sentence-transformers (MiniLM) |
| Vector Index | FAISS |
| Research API | OpenAlex |
| Frontend | React + TypeScript + Vite |

## 🏆 Hackathon Features

- ✅ Canonical JSON extraction
- ✅ OpenAlex enrichment  
- ✅ Comparative gap JSON
- ✅ Cursor-level trace
- 🔄 Novelty scoring (planned)
- 🔄 Reviewer simulation (planned)
- 🔄 Diagram generation (planned)

---

Built for **JournalSense** 🚀
