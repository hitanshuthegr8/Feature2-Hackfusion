from flask import Blueprint, jsonify
from services.grobid_service import GROBIDService
from utils.logging import setup_logger
import json

logger = setup_logger(__name__)
papers_bp = Blueprint('papers', __name__)

grobid_service = GROBIDService()

@papers_bp.route('/papers/<paper_id>', methods=['GET'])
def get_paper(paper_id: str):
    """
    Get GROBID-parsed JSON for a paper.
    
    Response:
        {
            "paper_id": "...",
            "title": "...",
            "abstract": "...",
            "sections": [...],
            "references": [...]
        }
    """
    try:
        logger.info(f"Fetching paper: {paper_id}")
        
        # Get JSON path
        json_path = grobid_service.get_json_path(paper_id)
        
        if not json_path.exists():
            return jsonify({
                'error': f'Paper not found: {paper_id}',
                'message': 'Paper has not been processed yet'
            }), 404
        
        # Load and return JSON
        with open(json_path, 'r', encoding='utf-8') as f:
            paper_data = json.load(f)
        
        paper_data['paper_id'] = paper_id
        
        return jsonify(paper_data), 200
    
    except Exception as e:
        logger.error(f"Get paper endpoint error: {e}")
        return jsonify({'error': str(e)}), 500
