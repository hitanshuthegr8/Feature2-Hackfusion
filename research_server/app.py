from flask import Flask, jsonify
from flask_cors import CORS
from config import Config
from routes.search import search_bp
from routes.papers import papers_bp
from routes.analysis import analysis_bp
from routes.pipeline import pipeline_bp
from utils.logging import setup_logger

logger = setup_logger(__name__)

def create_app():
    """Create and configure Flask application."""
    app = Flask(__name__)
    app.config.from_object(Config)
    
    # Enable CORS for all routes
    CORS(app)
    
    # Register blueprints
    app.register_blueprint(search_bp)
    app.register_blueprint(papers_bp)
    app.register_blueprint(analysis_bp)
    app.register_blueprint(pipeline_bp)
    
    # Global error handler
    @app.errorhandler(404)
    def not_found(error):
        return jsonify({'error': 'Endpoint not found'}), 404
    
    @app.errorhandler(500)
    def internal_error(error):
        logger.error(f"Internal server error: {error}")
        return jsonify({'error': 'Internal server error'}), 500
    
    # Health check endpoint
    @app.route('/health', methods=['GET'])
    def health():
        return jsonify({'status': 'healthy', 'service': 'research_server'}), 200
    
    # Root endpoint
    @app.route('/', methods=['GET'])
    def root():
        return jsonify({
            'service': 'Research Paper Analysis Server',
            'version': '2.0.0',
            'endpoints': {
                'POST /pipeline/analyze': '🚀 Full pipeline: Query → TF-IDF → OpenAlex → PDF → GROBID → LLM',
                'GET /pipeline/status': '📊 Check pipeline & service status',
                'POST /search': 'Search for papers (TF-IDF + OpenAlex)',
                'GET /papers/<paper_id>': 'Get GROBID-parsed paper JSON',
                'GET /analysis/<paper_id>': 'Get extracted models/datasets/baselines',
                'POST /analysis': 'Process multiple papers',
                'GET /health': 'Health check'
            }
        }), 200
    
    logger.info("Flask application created successfully")
    return app


if __name__ == '__main__':
    app = create_app()
    logger.info(f"Starting server on {Config.FLASK_HOST}:{Config.FLASK_PORT}")
    app.run(
        host=Config.FLASK_HOST,
        port=Config.FLASK_PORT,
        debug=Config.DEBUG
    )
