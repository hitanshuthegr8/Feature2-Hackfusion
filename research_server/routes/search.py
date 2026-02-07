from flask import Blueprint, request, jsonify
from services.tfidf_service import TFIDFService
from services.openalex_service import OpenAlexService
from utils.logging import setup_logger

from flask_cors import CORS

logger = setup_logger(__name__)
search_bp = Blueprint('search', __name__)
CORS(search_bp) # Enable CORS for this blueprint

tfidf_service = TFIDFService()
openalex_service = OpenAlexService()

@search_bp.route('/search', methods=['POST', 'OPTIONS'])
def search_papers():
    """
    Search for papers using TF-IDF and OpenAlex.
    """
    if request.method == 'OPTIONS':
        # Manual CORS Preflight Handling
        response = jsonify({})
        response.headers.add("Access-Control-Allow-Origin", "*")
        response.headers.add("Access-Control-Allow-Headers", "Content-Type,Authorization")
        response.headers.add("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
        return response, 200

    try:
        # Validate request
        data = request.get_json()
        if not data:
            return jsonify({'error': 'Request body must be JSON'}), 400
        
        query = data.get('query')
        if not query:
            return jsonify({'error': 'Missing required field: query'}), 400
        
        max_results = data.get('max_results', 25)
        
        logger.info(f"Search request: {query}")
        
        # Extract keywords with TF-IDF
        keywords_weighted = tfidf_service.extract_keywords(query)
        keywords = [kw[0] for kw in keywords_weighted]  # Just the words
        
        if not keywords:
            return jsonify({
                'query': query,
                'keywords': [],
                'papers': [],
                'message': 'No valid keywords extracted from query'
            }), 200
        
        # Search OpenAlex
        papers = openalex_service.search_papers(keywords, max_results)
        
        response = jsonify({
            'query': query,
            'keywords': keywords_weighted,
            'papers': papers,
            'total_results': len(papers)
        })
        response.headers.add("Access-Control-Allow-Origin", "*")
        return response, 200
    
    except Exception as e:
        logger.error(f"Search endpoint error: {e}")
        response = jsonify({'error': str(e)})
        response.headers.add("Access-Control-Allow-Origin", "*")
        return response, 500
