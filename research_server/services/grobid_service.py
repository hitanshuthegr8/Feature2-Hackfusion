import requests
from pathlib import Path
from typing import Optional, Dict
from bs4 import BeautifulSoup
from config import Config
from utils.logging import setup_logger
from utils.retry import retry_with_backoff

logger = setup_logger(__name__)

class GROBIDService:
    """Service for parsing PDFs using GROBID."""
    
    def __init__(self):
        """Initialize GROBID service."""
        self.base_url = Config.GROBID_URL
        self.timeout = Config.GROBID_TIMEOUT
        self.json_dir = Config.GROBID_JSON_DIR
        self.json_dir.mkdir(parents=True, exist_ok=True)
    
    def get_json_path(self, paper_id: str) -> Path:
        """Get the file path for a paper's parsed JSON."""
        return self.json_dir / f"{paper_id}.json"
    
    @retry_with_backoff(max_attempts=2, backoff_factor=3)
    def parse_pdf(self, pdf_path: Path, paper_id: str) -> Optional[Dict]:
        """
        Parse PDF using GROBID and convert to structured JSON.
        
        Args:
            pdf_path: Path to PDF file
            paper_id: Paper identifier
            
        Returns:
            Parsed paper data as dictionary or None if parsing failed
        """
        json_path = self.get_json_path(paper_id)
        
        # Skip if already parsed
        if json_path.exists():
            logger.info(f"GROBID JSON already exists: {json_path.name}")
            try:
                import json
                with open(json_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load cached JSON: {e}")
        
        logger.info(f"Parsing PDF with GROBID: {pdf_path.name}")
        
        try:
            # Call GROBID API
            url = f"{self.base_url}/api/processFulltextDocument"
            
            with open(pdf_path, 'rb') as f:
                files = {'input': f}
                response = requests.post(
                    url,
                    files=files,
                    timeout=self.timeout
                )
            
            response.raise_for_status()
            
            # Parse TEI XML response
            tei_xml = response.text
            parsed_data = self._parse_tei_xml(tei_xml)
            
            # Save to JSON
            import json
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(parsed_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Parsed and saved GROBID JSON: {json_path.name}")
            return parsed_data
        
        except requests.RequestException as e:
            logger.error(f"GROBID API request failed for {pdf_path.name}: {e}")
            return None
        except Exception as e:
            logger.error(f"Failed to parse PDF {pdf_path.name}: {e}")
            return None
    
    def _parse_tei_xml(self, xml_string: str) -> Dict:
        """
        Parse GROBID TEI XML and extract structured data.
        
        Args:
            xml_string: TEI XML string from GROBID
            
        Returns:
            Dictionary with extracted paper structure
        """
        soup = BeautifulSoup(xml_string, 'xml')
        
        # Extract title
        title_elem = soup.find('titleStmt')
        title = title_elem.find('title').get_text(strip=True) if title_elem else "Unknown"
        
        # Extract abstract
        abstract_elem = soup.find('abstract')
        abstract = abstract_elem.get_text(strip=True) if abstract_elem else ""
        
        # Extract sections
        sections = []
        for div in soup.find_all('div', {'xmlns': True}):
            head = div.find('head')
            if head:
                section_title = head.get_text(strip=True)
                paragraphs = [p.get_text(strip=True) for p in div.find_all('p')]
                section_text = ' '.join(paragraphs)
                
                if section_text:
                    sections.append({
                        'title': section_title,
                        'text': section_text
                    })
        
        # Extract references (for datasets/models mentioned)
        references = []
        for ref in soup.find_all('biblStruct')[:50]:  # Limit to 50 refs
            ref_text = ref.get_text(strip=True)
            if ref_text:
                references.append(ref_text)
        
        # Build full text from sections
        full_text = f"{title}\n\n{abstract}\n\n"
        for section in sections:
            full_text += f"{section['title']}\n{section['text']}\n\n"
        
        return {
            'title': title,
            'abstract': abstract,
            'sections': sections,
            'references': references,
            'full_text': full_text.strip()
        }
