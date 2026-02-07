export enum AppState {
  SPLASH = 'splash',
  ONBOARDING = 'onboarding',
  INPUT = 'input',
  LOADING = 'loading',
  RESULT = 'result',
  DASHBOARD = 'dashboard',
  COMPARE = 'compare'
}

export interface Message {
  id: string;
  sender: 'user' | 'bot';
  text: string;
  timestamp: Date;
}

export interface Abstract {
  background: string;
  methods: string;
  results: string;
  conclusion: string;
}

// Canonical Research JSON types
export interface EntityTrace {
  text: string;
  label: string;
  canonical: string;
  section: string;
  start: number;
  end: number;
  context: string;
}

export interface OpenAlexData {
  work_id: string | null;
  cited_by_count: number;
  publication_year: number | null;
  concepts: string[];
  trend_velocity: number;
  is_sota: boolean;
  benchmark_coverage?: Record<string, number>;
}

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
  entity_traces: EntityTrace[];
  openalex?: OpenAlexData;
}

export interface ComparisonResult {
  total_papers: number;
  architecture_distribution: Record<string, number>;
  dataset_distribution: Record<string, number>;
  task_distribution: Record<string, number>;
  baseline_distribution: Record<string, number>;
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