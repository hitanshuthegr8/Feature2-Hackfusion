import requests
from pathlib import Path
from typing import Optional
from config import Config
from utils.logging import setup_logger
from utils.hashing import hash_string
from utils.retry import retry_with_backoff

logger = setup_logger(__name__)

class PDFService:
    """Service for downloading and managing PDF files."""
    
    def __init__(self):
        """Initialize PDF service."""
        self.pdf_dir = Config.PDF_DIR
        self.pdf_dir.mkdir(parents=True, exist_ok=True)
        self.max_size_bytes = Config.PDF_MAX_SIZE_MB * 1024 * 1024
    
    def get_pdf_path(self, paper_id: str, pdf_url: str) -> Path:
        """
        Get the file path for a paper's PDF.
        
        Args:
            paper_id: Paper identifier
            pdf_url: URL of the PDF
            
        Returns:
            Path to PDF file
        """
        # Hash the URL to create a unique filename
        url_hash = hash_string(pdf_url)[:16]
        filename = f"{paper_id}_{url_hash}.pdf"
        return self.pdf_dir / filename
    
    @retry_with_backoff(max_attempts=3, backoff_factor=2)
    def download_pdf(self, paper_id: str, pdf_url: str) -> Optional[Path]:
        """
        Download PDF from URL and save to disk.
        
        Args:
            paper_id: Paper identifier
            pdf_url: URL of the PDF to download
            
        Returns:
            Path to downloaded PDF or None if download failed
        """
        pdf_path = self.get_pdf_path(paper_id, pdf_url)
        
        # Skip if already downloaded
        if pdf_path.exists():
            logger.info(f"PDF already exists: {pdf_path.name}")
            return pdf_path
        
        logger.info(f"Downloading PDF: {pdf_url}")
        
        try:
            # Stream download to handle large files
            response = requests.get(
                pdf_url,
                stream=True,
                timeout=Config.PDF_TIMEOUT,
                headers={'User-Agent': 'Mozilla/5.0'}
            )
            response.raise_for_status()
            
            # Validate content type
            content_type = response.headers.get('Content-Type', '')
            if 'pdf' not in content_type.lower():
                logger.warning(
                    f"Invalid content type: {content_type} for {pdf_url}"
                )
                # Continue anyway as some servers don't set proper headers
            
            # Check file size
            content_length = response.headers.get('Content-Length')
            if content_length and int(content_length) > self.max_size_bytes:
                logger.warning(
                    f"PDF too large ({int(content_length) / 1024 / 1024:.1f}MB): {pdf_url}"
                )
                return None
            
            # Download in chunks
            downloaded_size = 0
            with open(pdf_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded_size += len(chunk)
                        
                        # Enforce size limit during download
                        if downloaded_size > self.max_size_bytes:
                            logger.warning(
                                f"PDF exceeded size limit during download: {pdf_url}"
                            )
                            pdf_path.unlink()
                            return None
            
            logger.info(
                f"Downloaded PDF: {pdf_path.name} "
                f"({downloaded_size / 1024 / 1024:.1f}MB)"
            )
            return pdf_path
        
        except requests.RequestException as e:
            logger.error(f"Failed to download PDF from {pdf_url}: {e}")
            # Clean up partial download
            if pdf_path.exists():
                pdf_path.unlink()
            return None
        except IOError as e:
            logger.error(f"Failed to save PDF to {pdf_path}: {e}")
            if pdf_path.exists():
                pdf_path.unlink()
            return None
