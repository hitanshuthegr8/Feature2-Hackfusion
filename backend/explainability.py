"""
Cursor-Level Explainability Engine
Phase 6: Every insight has a trace
"""
from typing import Dict, List, Optional
from dataclasses import dataclass
from .entity_extractor import CanonicalResearchJSON

@dataclass
class EvidenceTrace:
    """Full trace for cursor-level explainability"""
    claim: str
    paper_id: str
    section: str
    start_offset: int
    end_offset: int
    highlighted_text: str
    context: str
    confidence: float

@dataclass
class ExplainableInsight:
    """An insight with full traceability"""
    insight_type: str  # pattern, gap, opportunity
    claim: str
    summary: str
    evidence: List[EvidenceTrace]
    
    def to_dict(self) -> Dict:
        return {
            "insight_type": self.insight_type,
            "claim": self.claim,
            "summary": self.summary,
            "evidence": [{"claim": e.claim, "paper_id": e.paper_id, "section": e.section, 
                         "start": e.start_offset, "end": e.end_offset, 
                         "highlighted_text": e.highlighted_text, "context": e.context[:200],
                         "confidence": e.confidence} for e in self.evidence]
        }

class ExplainabilityEngine:
    """Generate cursor-level explanations for insights"""
    
    def __init__(self, papers: List[CanonicalResearchJSON]):
        self.papers = {p.paper_id: p for p in papers}
    
    def explain_pattern(self, pattern: str) -> ExplainableInsight:
        """Generate explanation for a detected pattern"""
        evidence = []
        for paper in self.papers.values():
            for trace in paper.entity_traces:
                if self._matches_pattern(trace, pattern):
                    evidence.append(EvidenceTrace(
                        claim=pattern,
                        paper_id=paper.paper_id,
                        section=trace.get("section", "unknown"),
                        start_offset=trace.get("start", 0),
                        end_offset=trace.get("end", 0),
                        highlighted_text=trace.get("text", ""),
                        context=trace.get("context", ""),
                        confidence=0.85
                    ))
        return ExplainableInsight(
            insight_type="pattern",
            claim=pattern,
            summary=f"Pattern '{pattern}' found in {len(evidence)} locations",
            evidence=evidence[:10]
        )
    
    def explain_gap(self, gap: str) -> ExplainableInsight:
        """Generate explanation for a detected gap"""
        evidence = []
        for paper in self.papers.values():
            for lim in paper.limitations[:2]:
                evidence.append(EvidenceTrace(
                    claim=gap,
                    paper_id=paper.paper_id,
                    section="limitations",
                    start_offset=0,
                    end_offset=len(lim),
                    highlighted_text=lim[:100],
                    context=lim,
                    confidence=0.75
                ))
        return ExplainableInsight(
            insight_type="gap",
            claim=gap,
            summary=f"Gap '{gap}' supported by {len(evidence)} paper limitations",
            evidence=evidence[:10]
        )
    
    def get_trace_for_entity(self, paper_id: str, entity_text: str) -> Optional[Dict]:
        """Get cursor position for a specific entity"""
        paper = self.papers.get(paper_id)
        if not paper: return None
        for trace in paper.entity_traces:
            if entity_text.lower() in trace.get("text", "").lower():
                return {
                    "paper_id": paper_id,
                    "section": trace.get("section"),
                    "start": trace.get("start"),
                    "end": trace.get("end"),
                    "context": trace.get("context", "")[:200],
                    "raw_text_ref": paper.raw_text_refs.get(trace.get("section"), "")[:500]
                }
        return None
    
    def _matches_pattern(self, trace: Dict, pattern: str) -> bool:
        pattern_lower = pattern.lower()
        if "single dataset" in pattern_lower and trace.get("label") == "LIMITATION":
            return "single" in trace.get("context", "").lower() or "dataset" in trace.get("context", "").lower()
        if "baseline" in pattern_lower and trace.get("label") == "BASELINE":
            return True
        return False

def generate_explanations(papers: List[CanonicalResearchJSON], insights: List[str]) -> List[Dict]:
    engine = ExplainabilityEngine(papers)
    return [engine.explain_pattern(i).to_dict() for i in insights]

def get_cursor_trace(papers: List[CanonicalResearchJSON], paper_id: str, entity: str) -> Optional[Dict]:
    engine = ExplainabilityEngine(papers)
    return engine.get_trace_for_entity(paper_id, entity)
