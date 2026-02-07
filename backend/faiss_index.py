"""
FAISS Vector Indexing for Research Papers
Phase 4: Vectorization Done Right
"""
import numpy as np
import faiss
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging
from sentence_transformers import SentenceTransformer
from .config import EMBEDDING_MODEL, FAISS_DIMENSION, INDEX_DIR
from .entity_extractor import CanonicalResearchJSON

logger = logging.getLogger(__name__)

@dataclass
class VectorEntry:
    vector_id: str
    paper_id: str
    section: str
    embedding_text: str
    openalex_id: Optional[str]

class FAISSPaperIndex:
    def __init__(self, index_name: str = "papers"):
        self.index_name = index_name
        self.index_path = INDEX_DIR / f"{index_name}.index"
        self.metadata_path = INDEX_DIR / f"{index_name}_metadata.json"
        try:
            self.model = SentenceTransformer(EMBEDDING_MODEL)
            self.dimension = self.model.get_sentence_embedding_dimension()
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            self.model = None
            self.dimension = FAISS_DIMENSION
        self.index: Optional[faiss.Index] = None
        self.metadata: Dict[int, VectorEntry] = {}
        self.papers: Dict[str, CanonicalResearchJSON] = {}
        self._current_id = 0
        self._load_or_create_index()
    
    def _load_or_create_index(self):
        if self.index_path.exists() and self.metadata_path.exists():
            try:
                self.index = faiss.read_index(str(self.index_path))
                with open(self.metadata_path, 'r') as f:
                    data = json.load(f)
                    self.metadata = {int(k): VectorEntry(**v) for k, v in data.get("metadata", {}).items()}
                    self._current_id = data.get("current_id", 0)
            except Exception as e:
                logger.error(f"Failed to load index: {e}")
                self._create_new_index()
        else:
            self._create_new_index()
    
    def _create_new_index(self):
        self.index = faiss.IndexFlatIP(self.dimension)
        self.metadata = {}
        self._current_id = 0
    
    def save(self):
        INDEX_DIR.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(self.index_path))
        with open(self.metadata_path, 'w') as f:
            json.dump({"metadata": {k: {"vector_id": v.vector_id, "paper_id": v.paper_id, "section": v.section, "embedding_text": v.embedding_text, "openalex_id": v.openalex_id} for k, v in self.metadata.items()}, "current_id": self._current_id}, f)
    
    def _create_structured_summary(self, paper: CanonicalResearchJSON) -> str:
        parts = []
        if paper.architecture: parts.append(f"Architecture: {', '.join(paper.architecture)}")
        if paper.modules: parts.append(f"Modules: {', '.join(paper.modules)}")
        if paper.tasks: parts.append(f"Task: {', '.join(paper.tasks)}")
        if paper.datasets: parts.append(f"Dataset: {', '.join(paper.datasets)}")
        if paper.metrics: parts.append(f"Metrics: {', '.join([f'{k}: {v}' if v else k for k, v in paper.metrics.items()])}")
        if paper.baselines: parts.append(f"Baselines: {', '.join(paper.baselines)}")
        if paper.limitations: parts.append(f"Limitation: {paper.limitations[0]}")
        return "\n".join(parts)
    
    def _embed_text(self, text: str) -> np.ndarray:
        if self.model is None: raise RuntimeError("Embedding model not loaded")
        embedding = self.model.encode([text], convert_to_numpy=True)
        faiss.normalize_L2(embedding)
        return embedding[0]
    
    def add_paper(self, paper: CanonicalResearchJSON) -> List[str]:
        if self.model is None: raise RuntimeError("Embedding model not loaded")
        vector_ids = []
        self.papers[paper.paper_id] = paper
        summary = self._create_structured_summary(paper)
        if summary:
            vec_id = f"{paper.paper_id}_summary"
            embedding = self._embed_text(summary)
            self.index.add(embedding.reshape(1, -1))
            self.metadata[self._current_id] = VectorEntry(vec_id, paper.paper_id, "summary", summary, paper.openalex.get("work_id") if paper.openalex else None)
            vector_ids.append(vec_id)
            self._current_id += 1
        for section_name in ["methodology", "results", "abstract"]:
            if section_name in paper.raw_text_refs:
                section_text = paper.raw_text_refs[section_name][:1000]
                if section_text:
                    vec_id = f"{paper.paper_id}_{section_name}"
                    embedding = self._embed_text(section_text)
                    self.index.add(embedding.reshape(1, -1))
                    self.metadata[self._current_id] = VectorEntry(vec_id, paper.paper_id, section_name, section_text[:500], paper.openalex.get("work_id") if paper.openalex else None)
                    vector_ids.append(vec_id)
                    self._current_id += 1
        return vector_ids
    
    def search(self, query: str, k: int = 10, filter_section: Optional[str] = None) -> List[Tuple[VectorEntry, float, CanonicalResearchJSON]]:
        if self.model is None: raise RuntimeError("Embedding model not loaded")
        if self.index.ntotal == 0: return []
        query_embedding = self._embed_text(query)
        search_k = k * 3 if filter_section else k
        distances, indices = self.index.search(query_embedding.reshape(1, -1), min(search_k, self.index.ntotal))
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1: continue
            entry = self.metadata.get(idx)
            if entry is None: continue
            if filter_section and entry.section != filter_section: continue
            paper = self.papers.get(entry.paper_id)
            if paper: results.append((entry, float(dist), paper))
            if len(results) >= k: break
        return results
    
    def get_paper(self, paper_id: str) -> Optional[CanonicalResearchJSON]: return self.papers.get(paper_id)
    def get_all_papers(self) -> List[CanonicalResearchJSON]: return list(self.papers.values())
    def clear(self): self._create_new_index(); self.papers = {}; self.save()

_paper_index: Optional[FAISSPaperIndex] = None

def get_paper_index() -> FAISSPaperIndex:
    global _paper_index
    if _paper_index is None: _paper_index = FAISSPaperIndex()
    return _paper_index

def add_paper_to_index(paper: CanonicalResearchJSON) -> List[str]:
    index = get_paper_index()
    vector_ids = index.add_paper(paper)
    index.save()
    return vector_ids

def search_papers(query: str, k: int = 10) -> List[Dict]:
    index = get_paper_index()
    results = index.search(query, k)
    return [{"paper_id": paper.paper_id, "title": paper.title, "score": score, "section": entry.section, "architecture": paper.architecture, "datasets": paper.datasets, "tasks": paper.tasks} for entry, score, paper in results]
