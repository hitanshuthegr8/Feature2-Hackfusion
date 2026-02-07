"""
PDF Text Extraction and Section Segmentation
Phase 1.2: PDF → Text with section awareness
"""
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import fitz  # PyMuPDF


@dataclass
class TextSection:
    """Represents a section of extracted text with positioning info"""
    name: str
    content: str
    start_page: int
    end_page: int
    start_offset: int  # Character offset in full text
    end_offset: int
    sentences: List[Dict] = field(default_factory=list)  # For cursor-level explainability


@dataclass
class ExtractedDocument:
    """Complete extracted document with sections"""
    filename: str
    title: str
    full_text: str
    sections: Dict[str, TextSection]
    page_count: int
    word_count: int
    raw_text_refs: Dict[str, str]  # section_name -> raw text (for JSON)


# Common section headers in research papers
SECTION_PATTERNS = {
    'abstract': r'(?i)^abstract\s*$|^summary\s*$',
    'introduction': r'(?i)^(1\.?\s*)?introduction\s*$|^background\s*$',
    'related_work': r'(?i)^(2\.?\s*)?(related\s+work|literature\s+review|previous\s+work)\s*$',
    'methodology': r'(?i)^(3\.?\s*)?(method|methodology|materials?\s+(and|&)\s+methods?|approach|proposed\s+(method|approach))\s*$',
    'experiments': r'(?i)^(4\.?\s*)?(experiment|experimental\s+(setup|results?)|implementation)\s*$',
    'results': r'(?i)^(5\.?\s*)?(results?|findings|evaluation|performance)\s*$',
    'discussion': r'(?i)^(6\.?\s*)?(discussion|analysis)\s*$',
    'conclusion': r'(?i)^(7\.?\s*)?(conclusion|conclusions?|summary|future\s+work)\s*$',
    'references': r'(?i)^(references?|bibliography|works?\s+cited)\s*$',
}


class PDFExtractor:
    """
    Extracts text from PDF files with section segmentation.
    Provides sentence-level offsets for cursor explainability.
    """
    
    def __init__(self):
        self.section_patterns = {k: re.compile(v) for k, v in SECTION_PATTERNS.items()}
    
    def extract(self, pdf_path: str) -> ExtractedDocument:
        """
        Extract text from PDF with section segmentation.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            ExtractedDocument with full text and sections
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        
        # Extract raw text from all pages
        doc = fitz.open(str(pdf_path))
        pages_text: List[Tuple[int, str]] = []
        full_text_parts = []
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text("text")
            pages_text.append((page_num, text))
            full_text_parts.append(text)
        
        full_text = "\n".join(full_text_parts)
        doc.close()
        
        # Extract title (usually first non-empty line that's not a header)
        title = self._extract_title(pages_text)
        
        # Segment into sections
        sections = self._segment_sections(pages_text, full_text)
        
        # Build raw_text_refs for canonical JSON
        raw_text_refs = {
            name: section.content[:2000]  # First 2000 chars per section
            for name, section in sections.items()
        }
        
        return ExtractedDocument(
            filename=pdf_path.name,
            title=title,
            full_text=full_text,
            sections=sections,
            page_count=len(pages_text),
            word_count=len(full_text.split()),
            raw_text_refs=raw_text_refs
        )
    
    def _extract_title(self, pages_text: List[Tuple[int, str]]) -> str:
        """Extract paper title from first page"""
        if not pages_text:
            return "Unknown Title"
        
        first_page_text = pages_text[0][1]
        lines = [l.strip() for l in first_page_text.split('\n') if l.strip()]
        
        # Skip very short lines (likely headers/page numbers)
        for line in lines[:10]:  # Check first 10 non-empty lines
            if len(line) > 20 and len(line) < 300:
                # Skip if it matches a section header
                is_section = any(p.match(line) for p in self.section_patterns.values())
                if not is_section:
                    return line
        
        return lines[0] if lines else "Unknown Title"
    
    def _segment_sections(
        self, 
        pages_text: List[Tuple[int, str]], 
        full_text: str
    ) -> Dict[str, TextSection]:
        """Segment text into recognized sections with offsets"""
        sections: Dict[str, TextSection] = {}
        current_section: Optional[str] = None
        current_content: List[str] = []
        current_start_page: int = 0
        current_start_offset: int = 0
        
        offset = 0  # Track position in full_text
        
        for page_num, page_text in pages_text:
            lines = page_text.split('\n')
            
            for line in lines:
                stripped = line.strip()
                
                # Check if this line is a section header
                detected_section = self._detect_section(stripped)
                
                if detected_section and detected_section != current_section:
                    # Save previous section
                    if current_section and current_content:
                        content = '\n'.join(current_content)
                        sections[current_section] = TextSection(
                            name=current_section,
                            content=content,
                            start_page=current_start_page,
                            end_page=page_num,
                            start_offset=current_start_offset,
                            end_offset=offset,
                            sentences=self._extract_sentences(content, current_start_offset)
                        )
                    
                    # Start new section
                    current_section = detected_section
                    current_content = []
                    current_start_page = page_num
                    current_start_offset = offset
                    
                elif current_section:
                    current_content.append(line)
                
                offset += len(line) + 1  # +1 for newline
        
        # Save final section
        if current_section and current_content:
            content = '\n'.join(current_content)
            sections[current_section] = TextSection(
                name=current_section,
                content=content,
                start_page=current_start_page,
                end_page=len(pages_text) - 1,
                start_offset=current_start_offset,
                end_offset=offset,
                sentences=self._extract_sentences(content, current_start_offset)
            )
        
        # If no sections detected, create a "body" section
        if not sections:
            sections['body'] = TextSection(
                name='body',
                content=full_text,
                start_page=0,
                end_page=len(pages_text) - 1,
                start_offset=0,
                end_offset=len(full_text),
                sentences=self._extract_sentences(full_text, 0)
            )
        
        return sections
    
    def _detect_section(self, line: str) -> Optional[str]:
        """Detect if a line is a section header"""
        for section_name, pattern in self.section_patterns.items():
            if pattern.match(line):
                return section_name
        return None
    
    def _extract_sentences(self, text: str, base_offset: int) -> List[Dict]:
        """
        Extract sentences with their offsets for cursor-level explainability.
        
        Returns list of dicts with:
        - text: sentence text
        - start: absolute start offset
        - end: absolute end offset
        """
        sentences = []
        # Simple sentence splitting (can enhance with spaCy later)
        sentence_pattern = re.compile(r'(?<=[.!?])\s+(?=[A-Z])')
        
        current_pos = 0
        parts = sentence_pattern.split(text)
        
        for part in parts:
            if part.strip():
                start = text.find(part, current_pos)
                if start == -1:
                    start = current_pos
                end = start + len(part)
                
                sentences.append({
                    'text': part.strip(),
                    'start': base_offset + start,
                    'end': base_offset + end
                })
                current_pos = end
        
        return sentences


def extract_text_from_pdf(pdf_path: str) -> ExtractedDocument:
    """Convenience function for PDF extraction"""
    extractor = PDFExtractor()
    return extractor.extract(pdf_path)
