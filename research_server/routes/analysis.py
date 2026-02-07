from flask import Blueprint, jsonify, request
from services.extraction_service import ExtractionService
from services.openalex_service import OpenAlexService
from utils.logging import setup_logger

logger = setup_logger(__name__)
analysis_bp = Blueprint('analysis', __name__)

extraction_service = ExtractionService()
openalex_service = OpenAlexService()

@analysis_bp.route('/analysis/<paper_id>', methods=['GET'])
def analyze_paper(paper_id: str):
    """
    Get extracted models, datasets, and baselines for a paper.
    
    Response:
        {
            "paper_id": "...",
            "title": "...",
            "models": ["ResNet-50", ...],
            "datasets": ["ImageNet", ...],
            "baselines": [{"metric": "Accuracy", "value": "92.3%"}, ...]
        }
    """
    try:
        logger.info(f"Analyzing paper: {paper_id}")
        
        # Check if already processed
        from services.grobid_service import GROBIDService
        from services.ollama_service import OllamaService
        import json
        
        grobid_service = GROBIDService()
        ollama_service = OllamaService()
        
        json_path = grobid_service.get_json_path(paper_id)
        
        if not json_path.exists():
            return jsonify({
                'error': f'Paper not found: {paper_id}',
                'message': 'Paper has not been processed yet. Use POST /analysis first.'
            }), 404
        
        # Load GROBID data
        with open(json_path, 'r', encoding='utf-8') as f:
            grobid_data = json.load(f)
        
        # Extract with Ollama
        full_text = grobid_data.get('full_text', '')
        extracted_info = ollama_service.extract_structured_info(full_text)
        
        return jsonify({
            'paper_id': paper_id,
            'title': grobid_data.get('title'),
            'models': extracted_info.get('models', []),
            'datasets': extracted_info.get('datasets', []),
            'baselines': extracted_info.get('baselines', [])
        }), 200
    
    except Exception as e:
        logger.error(f"Analysis endpoint error: {e}")
        return jsonify({'error': str(e)}), 500


@analysis_bp.route('/analysis', methods=['POST'])
def process_papers():
    """
    Process multiple papers through the full extraction pipeline.
    
    Request body:
        {
            "papers": [
                {
                    "id": "W123456789",
                    "title": "...",
                    "pdf_url": "https://..."
                },
                ...
            ]
        }
    
    Response:
        {
            "processed": [...],
            "failed": [...]
        }
    """
    try:
        data = request.get_json()
        if not data or 'papers' not in data:
            return jsonify({'error': 'Missing required field: papers'}), 400
        
        papers = data['papers']
        
        if not isinstance(papers, list):
            return jsonify({'error': 'papers must be a list'}), 400
        
        logger.info(f"Processing {len(papers)} papers")
        
        processed = []
        failed = []
        
        for paper in papers:
            result = extraction_service.process_paper(paper)
            if result:
                processed.append({
                    'paper_id': result['paper_id'],
                    'title': result['title'],
                    'models': result['extracted_info']['models'],
                    'datasets': result['extracted_info']['datasets'],
                    'baselines': result['extracted_info']['baselines']
                })
            else:
                failed.append(paper.get('id', 'unknown'))
        
        return jsonify({
            'processed': processed,
            'failed': failed,
            'total': len(papers),
            'success_count': len(processed),
            'failure_count': len(failed)
        }), 200
    
    except Exception as e:
        logger.error(f"Process papers endpoint error: {e}")
        return jsonify({'error': str(e)}), 500
