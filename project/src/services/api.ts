/**
 * JournalSense API Client
 * Connects React frontend to Flask backend
 */

const API_BASE = 'http://localhost:5000';

export interface CanonicalPaper {
    paper_id: string;
    title: string;
    source: string;
    architecture: string[];
    modules: string[];
    datasets: string[];
    metrics: Record<string, number | null>;
    baselines: string[];
    tasks: string[];
    limitations: string[];
    contributions: string[];
    intent_phrases: string[];
    raw_text_refs: Record<string, string>;
    entity_traces: Array<{
        text: string;
        label: string;
        canonical: string;
        section: string;
        start: number;
        end: number;
        context: string;
    }>;
    openalex?: {
        work_id: string | null;
        cited_by_count: number;
        publication_year: number | null;
        concepts: string[];
        trend_velocity: number;
        is_sota: boolean;
    };
}

export interface ComparisonResult {
    total_papers: number;
    architecture_distribution: Record<string, number>;
    dataset_distribution: Record<string, number>;
    task_distribution: Record<string, number>;
    common_patterns: string[];
    missing_patterns: string[];
    overused_baselines: string[];
    novel_opportunities: string[];
    evaluation_gaps: string[];
    evidence: Array<{
        paper_id: string;
        claim: string;
        section: string;
        context: string;
    }>;
}

export interface CursorTrace {
    paper_id: string;
    section: string;
    start: number;
    end: number;
    context: string;
    raw_text_ref: string;
}

export interface SearchResult {
    paper_id: string;
    title: string;
    score: number;
    section: string;
    architecture: string[];
    datasets: string[];
    tasks: string[];
}

// Health check
export async function checkHealth(): Promise<{ status: string; indexed_papers: number; total_vectors: number }> {
    const res = await fetch(`${API_BASE}/health`);
    return res.json();
}

// Upload PDF
export async function uploadPDF(file: File): Promise<{ success: boolean; paper_id: string; canonical_json: CanonicalPaper }> {
    const formData = new FormData();
    formData.append('file', file);

    const res = await fetch(`${API_BASE}/upload-pdf`, {
        method: 'POST',
        body: formData
    });
    return res.json();
}

// Search by topic
export async function searchTopic(topic: string, limit: number = 25): Promise<{ success: boolean; results: any[] }> {
    const res = await fetch(`${API_BASE}/search-topic`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ topic, limit })
    });
    return res.json();
}

// Get all papers
export async function getAllPapers(): Promise<{ papers: CanonicalPaper[]; count: number }> {
    const res = await fetch(`${API_BASE}/papers`);
    return res.json();
}

// Get specific paper
export async function getPaper(paperId: string): Promise<{ paper: CanonicalPaper }> {
    const res = await fetch(`${API_BASE}/papers/${paperId}`);
    return res.json();
}

// Compare papers
export async function comparePapers(): Promise<{ success: boolean; analysis: ComparisonResult }> {
    const res = await fetch(`${API_BASE}/compare`);
    return res.json();
}

// Get cursor trace
export async function getCursorTrace(paperId: string, entity: string): Promise<{ success: boolean; trace: CursorTrace }> {
    const res = await fetch(`${API_BASE}/explain/${paperId}/${encodeURIComponent(entity)}`);
    return res.json();
}

// Semantic search
export async function searchPapers(query: string, k: number = 10): Promise<{ success: boolean; results: SearchResult[] }> {
    const res = await fetch(`${API_BASE}/search`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query, k })
    });
    return res.json();
}

// Clear index
export async function clearIndex(): Promise<{ success: boolean }> {
    const res = await fetch(`${API_BASE}/clear`, { method: 'POST' });
    return res.json();
}
