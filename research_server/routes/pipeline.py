"""
Full Research Paper Analysis Pipeline
=======================================
1. TF-IDF Keyword Extraction
2. OpenAlex Paper Search
3. PDF Download & Normalization
4. GROBID → JSON Parsing
5. LLM Extraction (Models, Datasets, Baselines)
"""

from flask import Blueprint, request, jsonify
from flask_cors import CORS
from services.tfidf_service import TFIDFService
from services.openalex_service import OpenAlexService
from services.extraction_service import ExtractionService
from services.comparison_service import ComparisonService
from utils.logging import setup_logger
import time

logger = setup_logger(__name__)
pipeline_bp = Blueprint('pipeline', __name__)
CORS(pipeline_bp)

# Initialize services
tfidf_service = TFIDFService()
openalex_service = OpenAlexService()
extraction_service = ExtractionService()
comparison_service = ComparisonService()


@pipeline_bp.route('/pipeline/analyze', methods=['POST', 'OPTIONS'])
def full_pipeline():
    """
    Full end-to-end pipeline:
    Query → TF-IDF → OpenAlex → PDF Download → GROBID → LLM Extraction
    
    Request:
        {
            "query": "transformer based medical image segmentation",
            "max_papers": 5,
            "skip_processing": false  // Optional: skip PDF/GROBID if just want search
        }
    
    Response:
        {
            "query": "...",
            "keywords": [...],
            "papers": [
                {
                    "id": "W...",
                    "title": "...",
                    "authors": [...],
                    "year": 2023,
                    "pdf_url": "...",
                    "doi": "...",
                    "analysis": {
                        "models": ["ResNet-50", "U-Net"],
                        "datasets": ["ImageNet", "BraTS"],
                        "baselines": [{"metric": "Accuracy", "value": "95.2%"}]
                    },
                    "status": "success" | "no_pdf" | "grobid_failed" | "extraction_failed"
                }
            ],
            "stats": {
                "total_searched": 25,
                "total_processed": 5,
                "successful": 3,
                "failed": 2,
                "processing_time_seconds": 45.2
            }
        }
    """
    if request.method == 'OPTIONS':
        response = jsonify({})
        response.headers.add("Access-Control-Allow-Origin", "*")
        response.headers.add("Access-Control-Allow-Headers", "Content-Type")
        response.headers.add("Access-Control-Allow-Methods", "POST,OPTIONS")
        return response, 200

    start_time = time.time()
    
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'Request body must be JSON'}), 400
        
        query = data.get('query')
        if not query:
            return jsonify({'error': 'Missing required field: query'}), 400
        
        max_papers = min(data.get('max_papers', 5), 10)  # Cap at 10
        skip_processing = data.get('skip_processing', False)
        
        logger.info(f"🚀 Pipeline started: '{query}' (max_papers={max_papers})")
        
        # ===== Step 1: TF-IDF Keyword Extraction =====
        logger.info("📝 Step 1: Extracting keywords with TF-IDF...")
        keywords_weighted = tfidf_service.extract_keywords(query)
        keywords = [kw[0] for kw in keywords_weighted]
        
        if not keywords:
            return jsonify({
                'query': query,
                'keywords': [],
                'papers': [],
                'error': 'No valid keywords extracted from query'
            }), 200
        
        logger.info(f"   Keywords: {keywords}")
        
        # ===== Step 2: OpenAlex Search =====
        logger.info("🔍 Step 2: Searching OpenAlex for papers...")
        all_papers = openalex_service.search_papers(keywords, max_results=25)
        
        # Filter papers with PDF URLs and limit
        papers_with_pdf = [p for p in all_papers if p.get('pdf_url')][:max_papers]
        papers_no_pdf = [p for p in all_papers if not p.get('pdf_url')]
        
        logger.info(f"   Found {len(all_papers)} papers, {len(papers_with_pdf)} have PDFs")
        
        if skip_processing:
            # Just return search results
            return jsonify({
                'query': query,
                'keywords': keywords_weighted,
                'papers': all_papers,
                'stats': {
                    'total_searched': len(all_papers),
                    'with_pdf': len(papers_with_pdf),
                    'without_pdf': len(papers_no_pdf),
                    'processing_time_seconds': round(time.time() - start_time, 2)
                }
            }), 200
        
        # ===== Steps 3-5: Process each paper =====
        results = []
        successful = 0
        failed = 0
        
        for i, paper in enumerate(papers_with_pdf):
            paper_id = paper.get('id')
            logger.info(f"📄 Processing paper {i+1}/{len(papers_with_pdf)}: {paper_id}")
            
            try:
                # Full pipeline: PDF → GROBID → Ollama
                result = extraction_service.process_paper(paper)
                
                if result:
                    results.append({
                        'id': paper_id,
                        'title': paper.get('title'),
                        'authors': paper.get('authors', []),
                        'year': paper.get('year'),
                        'pdf_url': paper.get('pdf_url'),
                        'doi': paper.get('doi'),
                        'venue': paper.get('venue'),
                        'cited_by_count': paper.get('cited_by_count', 0),
                        'abstract': paper.get('abstract'),
                        'analysis': {
                            'models': result['extracted_info'].get('models', []),
                            'datasets': result['extracted_info'].get('datasets', []),
                            'baselines': result['extracted_info'].get('baselines', [])
                        },
                        'status': 'success'
                    })
                    successful += 1
                    logger.info(f"   ✅ Success: {len(result['extracted_info'].get('models', []))} models, "
                               f"{len(result['extracted_info'].get('datasets', []))} datasets")
                else:
                    results.append({
                        'id': paper_id,
                        'title': paper.get('title'),
                        'authors': paper.get('authors', []),
                        'year': paper.get('year'),
                        'pdf_url': paper.get('pdf_url'),
                        'status': 'processing_failed'
                    })
                    failed += 1
                    logger.warning(f"   ❌ Failed to process")
                    
            except Exception as e:
                logger.error(f"   ❌ Error processing paper {paper_id}: {e}")
                results.append({
                    'id': paper_id,
                    'title': paper.get('title'),
                    'status': 'error',
                    'error': str(e)
                })
                failed += 1
        
        # Add papers without PDFs as "no_pdf" status
        for paper in papers_no_pdf[:5]:  # Include up to 5 no-pdf papers for reference
            results.append({
                'id': paper.get('id'),
                'title': paper.get('title'),
                'authors': paper.get('authors', []),
                'year': paper.get('year'),
                'doi': paper.get('doi'),
                'status': 'no_pdf'
            })
        
        processing_time = round(time.time() - start_time, 2)
        logger.info(f"Pipeline complete: {successful}/{len(papers_with_pdf)} papers processed in {processing_time}s")
        
        # ===== Step 6: Compare & Aggregate Results =====
        logger.info("Generating comparison insights...")
        comparison = comparison_service.compare_papers(results)
        
        response = jsonify({
            'query': query,
            'keywords': keywords_weighted,
            'papers': results,
            'comparison': comparison,
            'stats': {
                'total_searched': len(all_papers),
                'total_processed': len(papers_with_pdf),
                'successful': successful,
                'failed': failed,
                'no_pdf_available': len(papers_no_pdf),
                'processing_time_seconds': processing_time
            }
        })
        response.headers.add("Access-Control-Allow-Origin", "*")
        return response, 200
        
    except Exception as e:
        logger.error(f"Pipeline error: {e}")
        import traceback
        traceback.print_exc()
        response = jsonify({'error': str(e)})
        response.headers.add("Access-Control-Allow-Origin", "*")
        return response, 500


@pipeline_bp.route('/pipeline/status', methods=['GET'])
def pipeline_status():
    """Check pipeline status and service health."""
    import requests
    from config import Config
    
    status = {
        'pipeline': 'ready',
        'services': {}
    }
    
    # Check GROBID
    try:
        r = requests.get(f"{Config.GROBID_URL}/api/isalive", timeout=5)
        status['services']['grobid'] = 'online' if r.status_code == 200 else 'error'
        status['services']['grobid_url'] = Config.GROBID_URL
    except:
        status['services']['grobid'] = 'offline'
    
    # Check Ollama
    try:
        r = requests.get(f"{Config.OLLAMA_URL}/api/tags", timeout=5)
        if r.status_code == 200:
            models = r.json().get('models', [])
            status['services']['ollama'] = 'online'
            status['services']['ollama_models'] = [m['name'] for m in models]
        else:
            status['services']['ollama'] = 'error'
    except:
        status['services']['ollama'] = 'offline'
    
    # Check OpenAlex (just a simple ping)
    status['services']['openalex'] = 'online'  # Public API, always available
    
    return jsonify(status), 200
