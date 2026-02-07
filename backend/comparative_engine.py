"""
Comparative Analysis Engine
Phase 5: Rule-based intelligence for gap discovery
"""
from typing import Dict, List, Optional, Set
from dataclasses import dataclass, field
from collections import Counter, defaultdict
from .entity_extractor import CanonicalResearchJSON

@dataclass
class ComparisonResult:
    """Machine-readable gap discovery output"""
    total_papers: int
    architecture_distribution: Dict[str, float]
    dataset_distribution: Dict[str, float]
    task_distribution: Dict[str, float]
    baseline_distribution: Dict[str, float]
    common_patterns: List[str]
    missing_patterns: List[str]
    overused_baselines: List[str]
    novel_opportunities: List[str]
    evaluation_gaps: List[str]
    evidence: List[Dict]  # Cursor-level traces
    
    def to_dict(self) -> Dict:
        return {
            "total_papers": self.total_papers,
            "architecture_distribution": self.architecture_distribution,
            "dataset_distribution": self.dataset_distribution,
            "task_distribution": self.task_distribution,
            "baseline_distribution": self.baseline_distribution,
            "common_patterns": self.common_patterns,
            "missing_patterns": self.missing_patterns,
            "overused_baselines": self.overused_baselines,
            "novel_opportunities": self.novel_opportunities,
            "evaluation_gaps": self.evaluation_gaps,
            "evidence": self.evidence
        }

class ComparativeAnalyzer:
    """Rule-based analysis across multiple papers"""
    
    def __init__(self, papers: List[CanonicalResearchJSON]):
        self.papers = papers
        self.n = len(papers)
    
    def analyze(self) -> ComparisonResult:
        if self.n == 0:
            return self._empty_result()
        
        arch_counter = Counter()
        dataset_counter = Counter()
        task_counter = Counter()
        baseline_counter = Counter()
        limitation_texts = []
        evidence = []
        
        for paper in self.papers:
            for arch in paper.architecture: arch_counter[arch] += 1
            for ds in paper.datasets: dataset_counter[ds] += 1
            for task in paper.tasks: task_counter[task] += 1
            for baseline in paper.baselines: baseline_counter[baseline] += 1
            for lim in paper.limitations:
                limitation_texts.append({"paper_id": paper.paper_id, "text": lim})
        
        arch_dist = {k: v / self.n for k, v in arch_counter.items()}
        dataset_dist = {k: v / self.n for k, v in dataset_counter.items()}
        task_dist = {k: v / self.n for k, v in task_counter.items()}
        baseline_dist = {k: v / self.n for k, v in baseline_counter.items()}
        
        common_patterns = self._find_common_patterns(limitation_texts)
        missing_patterns = self._find_missing_patterns(arch_counter, dataset_counter, task_counter)
        overused = [b for b, c in baseline_counter.items() if c / self.n > 0.5]
        novel = self._suggest_opportunities(arch_dist, dataset_dist, task_dist, common_patterns)
        gaps = self._find_evaluation_gaps(dataset_counter, task_counter)
        
        for paper in self.papers[:5]:
            for trace in paper.entity_traces[:3]:
                evidence.append({"paper_id": paper.paper_id, "claim": trace.get("label"), "section": trace.get("section"), "context": trace.get("context", "")[:150]})
        
        return ComparisonResult(
            total_papers=self.n,
            architecture_distribution=arch_dist,
            dataset_distribution=dataset_dist,
            task_distribution=task_dist,
            baseline_distribution=baseline_dist,
            common_patterns=common_patterns,
            missing_patterns=missing_patterns,
            overused_baselines=overused,
            novel_opportunities=novel,
            evaluation_gaps=gaps,
            evidence=evidence
        )
    
    def _find_common_patterns(self, limitations: List[Dict]) -> List[str]:
        patterns = []
        single_dataset = sum(1 for l in limitations if "single dataset" in l["text"].lower())
        if single_dataset / max(1, len(limitations)) > 0.3:
            patterns.append("single dataset evaluation")
        limited_domain = sum(1 for l in limitations if "domain" in l["text"].lower() or "limited" in l["text"].lower())
        if limited_domain / max(1, len(limitations)) > 0.3:
            patterns.append("limited domain generalization")
        return patterns if patterns else ["diverse evaluation approaches"]
    
    def _find_missing_patterns(self, arch: Counter, dataset: Counter, task: Counter) -> List[str]:
        missing = []
        if len(dataset) <= 2: missing.append("cross-dataset evaluation")
        if "cross_attention" not in arch and "self_attention" in arch: missing.append("cross-attention mechanisms")
        return missing if missing else ["comprehensive coverage detected"]
    
    def _suggest_opportunities(self, arch: Dict, dataset: Dict, task: Dict, patterns: List[str]) -> List[str]:
        opportunities = []
        if "single dataset evaluation" in patterns: opportunities.append("multi-dataset benchmarking")
        if max(arch.values(), default=0) > 0.7: opportunities.append("architecture diversity exploration")
        if "segmentation" in task and "vision_transformer" in arch: opportunities.append("efficient transformer variants")
        return opportunities if opportunities else ["current approaches well-balanced"]
    
    def _find_evaluation_gaps(self, dataset: Counter, task: Counter) -> List[str]:
        gaps = []
        if len(dataset) == 1: gaps.append("single benchmark dependency")
        if not any("cross" in t for t in task.keys()): gaps.append("no cross-domain evaluation")
        return gaps if gaps else ["evaluation coverage adequate"]
    
    def _empty_result(self) -> ComparisonResult:
        return ComparisonResult(0, {}, {}, {}, {}, [], [], [], [], [], [])

def analyze_papers(papers: List[CanonicalResearchJSON]) -> Dict:
    analyzer = ComparativeAnalyzer(papers)
    return analyzer.analyze().to_dict()
