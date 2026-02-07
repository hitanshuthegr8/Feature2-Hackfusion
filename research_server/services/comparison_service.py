"""
Comparison Service
==================
Aggregates and compares extracted information across multiple papers.
Provides intelligent insights on:
- Most common model architectures
- Popular datasets
- Performance baselines comparison
- Research trends
"""

from typing import Dict, List, Any
from collections import Counter, defaultdict
import re
from utils.logging import setup_logger

logger = setup_logger(__name__)


class ComparisonService:
    """Service for comparing and aggregating research paper analysis."""
    
    # Known model architecture categories
    MODEL_CATEGORIES = {
        'Transformer': ['transformer', 'bert', 'gpt', 'vit', 'swin', 'deit', 'attention'],
        'CNN': ['cnn', 'resnet', 'vgg', 'efficientnet', 'densenet', 'inception', 'mobilenet', 'convnet'],
        'U-Net': ['unet', 'u-net', 'attention u-net', 'unet++'],
        'GAN': ['gan', 'generative adversarial', 'discriminator', 'generator'],
        'RNN/LSTM': ['rnn', 'lstm', 'gru', 'recurrent'],
        'Hybrid': ['hybrid', 'transunet', 'swinunet', 'transfuse']
    }
    
    # Known dataset categories
    DATASET_CATEGORIES = {
        'Medical Imaging': ['brats', 'isic', 'covid', 'chest x-ray', 'ct scan', 'mri', 'dermatology', 'polyp'],
        'Natural Images': ['imagenet', 'coco', 'pascal voc', 'cityscapes', 'ade20k'],
        'Video': ['kinetics', 'ucf101', 'hmdb51', 'youtube'],
        'NLP': ['glue', 'squad', 'wikitext', 'imdb'],
        'Segmentation': ['pascal', 'ade20k', 'cityscapes', 'coco-stuff']
    }
    
    # Common metrics with normalization
    METRIC_PATTERNS = {
        'accuracy': r'(\d+\.?\d*)\s*%?',
        'f1': r'(\d+\.?\d*)',
        'iou': r'(\d+\.?\d*)\s*%?',
        'miou': r'(\d+\.?\d*)\s*%?',
        'dice': r'(\d+\.?\d*)\s*%?',
        'auc': r'(\d+\.?\d*)',
        'precision': r'(\d+\.?\d*)\s*%?',
        'recall': r'(\d+\.?\d*)\s*%?',
        'bleu': r'(\d+\.?\d*)',
        'rouge': r'(\d+\.?\d*)'
    }

    def compare_papers(self, papers: List[Dict]) -> Dict[str, Any]:
        """
        Generate comprehensive comparison across all analyzed papers.
        
        Args:
            papers: List of paper analysis results
            
        Returns:
            Comparison summary with aggregated insights
        """
        logger.info(f"Comparing {len(papers)} papers")
        
        # Filter successfully analyzed papers
        analyzed = [p for p in papers if p.get('status') == 'success' and p.get('analysis')]
        
        if not analyzed:
            return {
                'summary': 'No papers were successfully analyzed',
                'models': {},
                'datasets': {},
                'baselines': {},
                'insights': []
            }
        
        comparison = {
            'papers_analyzed': len(analyzed),
            'papers_total': len(papers),
            'models': self._aggregate_models(analyzed),
            'datasets': self._aggregate_datasets(analyzed),
            'baselines': self._aggregate_baselines(analyzed),
            'paper_details': self._build_paper_matrix(analyzed),
            'insights': [],
            'recommendations': []
        }
        
        # Generate intelligent insights
        comparison['insights'] = self._generate_insights(comparison)
        comparison['recommendations'] = self._generate_recommendations(comparison)
        
        return comparison
    
    def _aggregate_models(self, papers: List[Dict]) -> Dict:
        """Aggregate and categorize models across papers."""
        all_models = []
        model_by_paper = {}
        
        for paper in papers:
            models = paper.get('analysis', {}).get('models', [])
            paper_id = paper.get('id', 'unknown')
            model_by_paper[paper_id] = models
            all_models.extend(models)
        
        # Count occurrences
        model_counts = Counter(all_models)
        
        # Categorize models
        categorized = defaultdict(list)
        for model in all_models:
            model_lower = model.lower()
            categorized_flag = False
            for category, keywords in self.MODEL_CATEGORIES.items():
                if any(kw in model_lower for kw in keywords):
                    categorized[category].append(model)
                    categorized_flag = True
                    break
            if not categorized_flag:
                categorized['Other'].append(model)
        
        # Get unique models per category with counts
        category_summary = {}
        for cat, models in categorized.items():
            unique_models = list(set(models))
            category_summary[cat] = {
                'models': unique_models,
                'count': len(models),
                'unique_count': len(unique_models)
            }
        
        return {
            'total_mentions': len(all_models),
            'unique_models': list(set(all_models)),
            'frequency': dict(model_counts.most_common(15)),
            'by_category': category_summary,
            'by_paper': model_by_paper,
            'top_model': model_counts.most_common(1)[0] if model_counts else None
        }
    
    def _aggregate_datasets(self, papers: List[Dict]) -> Dict:
        """Aggregate and categorize datasets across papers."""
        all_datasets = []
        dataset_by_paper = {}
        
        for paper in papers:
            datasets = paper.get('analysis', {}).get('datasets', [])
            paper_id = paper.get('id', 'unknown')
            dataset_by_paper[paper_id] = datasets
            all_datasets.extend(datasets)
        
        # Count occurrences
        dataset_counts = Counter(all_datasets)
        
        # Categorize datasets
        categorized = defaultdict(list)
        for dataset in all_datasets:
            dataset_lower = dataset.lower()
            categorized_flag = False
            for category, keywords in self.DATASET_CATEGORIES.items():
                if any(kw in dataset_lower for kw in keywords):
                    categorized[category].append(dataset)
                    categorized_flag = True
                    break
            if not categorized_flag:
                categorized['Other'].append(dataset)
        
        category_summary = {}
        for cat, datasets in categorized.items():
            unique_datasets = list(set(datasets))
            category_summary[cat] = {
                'datasets': unique_datasets,
                'count': len(datasets),
                'unique_count': len(unique_datasets)
            }
        
        return {
            'total_mentions': len(all_datasets),
            'unique_datasets': list(set(all_datasets)),
            'frequency': dict(dataset_counts.most_common(15)),
            'by_category': category_summary,
            'by_paper': dataset_by_paper,
            'top_dataset': dataset_counts.most_common(1)[0] if dataset_counts else None
        }
    
    def _aggregate_baselines(self, papers: List[Dict]) -> Dict:
        """Aggregate and compare baselines across papers."""
        all_baselines = []
        baseline_by_paper = {}
        metric_values = defaultdict(list)
        
        for paper in papers:
            baselines = paper.get('analysis', {}).get('baselines', [])
            paper_id = paper.get('id', 'unknown')
            paper_title = paper.get('title', 'Unknown')[:50]
            baseline_by_paper[paper_id] = {
                'title': paper_title,
                'baselines': baselines
            }
            
            for baseline in baselines:
                metric = baseline.get('metric', '').lower().strip()
                value = baseline.get('value', '')
                
                if metric and value:
                    all_baselines.append(baseline)
                    # Try to parse numeric value
                    parsed_value = self._parse_metric_value(value)
                    if parsed_value is not None:
                        metric_values[metric].append({
                            'value': parsed_value,
                            'raw': value,
                            'paper_id': paper_id,
                            'paper_title': paper_title
                        })
        
        # Compute statistics for each metric
        metric_stats = {}
        for metric, values in metric_values.items():
            if values:
                numeric_vals = [v['value'] for v in values]
                metric_stats[metric] = {
                    'count': len(values),
                    'min': min(numeric_vals),
                    'max': max(numeric_vals),
                    'avg': sum(numeric_vals) / len(numeric_vals),
                    'best': max(values, key=lambda x: x['value']),
                    'values': values
                }
        
        return {
            'total_baselines': len(all_baselines),
            'metrics_found': list(metric_values.keys()),
            'metric_statistics': metric_stats,
            'by_paper': baseline_by_paper,
            'comparison_table': self._build_baseline_table(metric_values)
        }
    
    def _parse_metric_value(self, value: str) -> float:
        """Parse numeric value from metric string."""
        try:
            # Remove common non-numeric characters
            cleaned = re.sub(r'[^\d.]', '', str(value))
            if cleaned:
                return float(cleaned)
        except:
            pass
        return None
    
    def _build_baseline_table(self, metric_values: Dict) -> List[Dict]:
        """Build comparison table for baselines."""
        table = []
        for metric, values in metric_values.items():
            for entry in sorted(values, key=lambda x: x['value'], reverse=True)[:5]:
                table.append({
                    'metric': metric.title(),
                    'value': entry['raw'],
                    'numeric': entry['value'],
                    'paper': entry['paper_title']
                })
        return table
    
    def _build_paper_matrix(self, papers: List[Dict]) -> List[Dict]:
        """Build matrix showing what each paper uses."""
        matrix = []
        for paper in papers:
            analysis = paper.get('analysis', {})
            matrix.append({
                'id': paper.get('id'),
                'title': paper.get('title', '')[:60],
                'year': paper.get('year'),
                'citations': paper.get('cited_by_count', 0),
                'models': analysis.get('models', []),
                'datasets': analysis.get('datasets', []),
                'baselines': analysis.get('baselines', []),
                'model_count': len(analysis.get('models', [])),
                'dataset_count': len(analysis.get('datasets', [])),
                'baseline_count': len(analysis.get('baselines', []))
            })
        return sorted(matrix, key=lambda x: x['citations'], reverse=True)
    
    def _generate_insights(self, comparison: Dict) -> List[str]:
        """Generate human-readable insights from comparison data."""
        insights = []
        
        # Model insights
        models = comparison.get('models', {})
        if models.get('top_model'):
            name, count = models['top_model']
            insights.append(f"🏆 **{name}** is the most frequently mentioned model ({count} papers)")
        
        if models.get('by_category'):
            top_category = max(models['by_category'].items(), 
                             key=lambda x: x[1]['count'], default=(None, None))
            if top_category[0]:
                insights.append(f"🔬 **{top_category[0]}** architectures dominate this research area")
        
        # Dataset insights
        datasets = comparison.get('datasets', {})
        if datasets.get('top_dataset'):
            name, count = datasets['top_dataset']
            insights.append(f"📊 **{name}** is the benchmark dataset of choice ({count} papers)")
        
        # Baseline insights
        baselines = comparison.get('baselines', {})
        metric_stats = baselines.get('metric_statistics', {})
        
        for metric, stats in metric_stats.items():
            if stats.get('best'):
                best = stats['best']
                insights.append(
                    f"📈 Best **{metric.title()}**: {best['raw']} "
                    f"(from \"{best['paper_title']}\")"
                )
        
        # Paper coverage insights
        paper_details = comparison.get('paper_details', [])
        if paper_details:
            total_citations = sum(p['citations'] for p in paper_details)
            if total_citations > 0:
                insights.append(f"📚 Combined citation impact: **{total_citations:,}** citations")
        
        return insights
    
    def _generate_recommendations(self, comparison: Dict) -> List[str]:
        """Generate actionable recommendations based on comparison."""
        recommendations = []
        
        models = comparison.get('models', {})
        datasets = comparison.get('datasets', {})
        baselines = comparison.get('baselines', {})
        
        # Model recommendations
        if models.get('by_category'):
            categories = models['by_category']
            if 'Transformer' in categories and 'CNN' in categories:
                recommendations.append(
                    "💡 Consider **hybrid Transformer-CNN** architectures - "
                    "papers show both are popular in this domain"
                )
            elif 'Transformer' in categories:
                recommendations.append(
                    "💡 **Transformer-based** models are trending - "
                    "consider Vision Transformers or Swin Transformer"
                )
        
        # Dataset recommendations
        if datasets.get('unique_datasets'):
            unique_ds = datasets['unique_datasets']
            if len(unique_ds) >= 3:
                recommendations.append(
                    f"💡 Test on multiple datasets ({', '.join(unique_ds[:3])}) "
                    "for robust validation"
                )
        
        # Baseline recommendations
        metric_stats = baselines.get('metric_statistics', {})
        if metric_stats:
            primary_metrics = list(metric_stats.keys())[:2]
            if primary_metrics:
                recommendations.append(
                    f"💡 Focus on **{', '.join(m.title() for m in primary_metrics)}** "
                    "as key evaluation metrics"
                )
        
        return recommendations
    
    def generate_summary_text(self, comparison: Dict) -> str:
        """Generate a natural language summary of the comparison."""
        papers_analyzed = comparison.get('papers_analyzed', 0)
        
        if papers_analyzed == 0:
            return "No papers were successfully analyzed for comparison."
        
        models = comparison.get('models', {})
        datasets = comparison.get('datasets', {})
        
        summary_parts = [
            f"📊 **Analysis Summary** ({papers_analyzed} papers analyzed)",
            ""
        ]
        
        # Models summary
        if models.get('unique_models'):
            summary_parts.append(
                f"**Models Found:** {len(models['unique_models'])} unique architectures"
            )
            if models.get('frequency'):
                top_3 = list(models['frequency'].items())[:3]
                summary_parts.append(
                    f"  Top models: {', '.join(f'{m} ({c})' for m, c in top_3)}"
                )
        
        # Datasets summary
        if datasets.get('unique_datasets'):
            summary_parts.append(
                f"**Datasets Found:** {len(datasets['unique_datasets'])} benchmarks"
            )
            if datasets.get('frequency'):
                top_3 = list(datasets['frequency'].items())[:3]
                summary_parts.append(
                    f"  Top datasets: {', '.join(f'{d} ({c})' for d, c in top_3)}"
                )
        
        return "\n".join(summary_parts)
