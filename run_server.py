"""
Run script for JournalSense Backend
"""
import sys
import subprocess

def install_spacy_model():
    """Install spaCy model if not present"""
    try:
        import spacy
        spacy.load("en_core_web_sm")
    except:
        print("Installing spaCy model...")
        subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])

if __name__ == '__main__':
    install_spacy_model()
    from backend.app import app
    print("\n" + "="*60)
    print("[*] JournalSense API Server")
    print("="*60)
    print("\nEndpoints:")
    print("  POST /upload-pdf    - Upload and process PDF")
    print("  POST /search-topic  - Search by research topic")
    print("  GET  /papers        - Get all indexed papers")
    print("  GET  /papers/<id>   - Get specific paper")
    print("  GET  /compare       - Compare all papers (gap analysis)")
    print("  GET  /explain/<id>/<entity> - Get cursor trace")
    print("  POST /search        - Semantic search")
    print("\n" + "="*60 + "\n")
    app.run(host='0.0.0.0', port=5000, debug=True)
