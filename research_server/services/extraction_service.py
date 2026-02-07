from typing import Dict, Optional
from pathlib import Path
from services.pdf_service import PDFService
from services.grobid_service import GROBIDService
from services.ollama_service import OllamaService
from utils.logging import setup_logger

logger = setup_logger(__name__)

class ExtractionService:
    """Orchestrates the full extraction pipeline."""
    
    def __init__(self):
        """Initialize extraction service with sub-services."""
        self.pdf_service = PDFService()
        self.grobid_service = GROBIDService()
        self.ollama_service = OllamaService()
    
    def process_paper(self, paper: Dict) -> Optional[Dict]:
        """
        Process a paper through the full extraction pipeline.
        
        Args:
            paper: Paper metadata dictionary with pdf_url
            
        Returns:
            Dictionary with extracted information or None if processing failed
        """
        paper_id = paper.get('id')
        pdf_url = paper.get('pdf_url')
        
        if not paper_id or not pdf_url:
            logger.warning("Paper missing required fields (id or pdf_url)")
            return None
        
        logger.info(f"Processing paper: {paper_id}")
        
        # Step 1: Download PDF
        pdf_path = self.pdf_service.download_pdf(paper_id, pdf_url)
        if not pdf_path:
            logger.warning(f"Failed to download PDF for paper: {paper_id}")
            return None
        
        # Step 2: Parse with GROBID
        grobid_data = self.grobid_service.parse_pdf(pdf_path, paper_id)
        if not grobid_data:
            logger.warning(f"Failed to parse PDF with GROBID: {paper_id}")
            return None
        
        # Step 3: Extract with Ollama
        full_text = grobid_data.get('full_text', '')
        if not full_text:
            logger.warning(f"No full text available for extraction: {paper_id}")
            return None
        
        extracted_info = self.ollama_service.extract_structured_info(full_text)
        
        # Combine results
        result = {
            'paper_id': paper_id,
            'title': paper.get('title'),
            'authors': paper.get('authors'),
            'year': paper.get('year'),
            'grobid_data': grobid_data,
            'extracted_info': extracted_info
        }
        
        logger.info(f"Successfully processed paper: {paper_id}")
        return result
