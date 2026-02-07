
import os
import sys
import fitz
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Mock valid PDF creation
def create_test_pdf():
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 50), "Title: Test Paper on Vision Transformers\n\nAbstract\nThis paper proposes a new Vision Transformer (ViT) architecture for medical image segmentation using the BraTS 2021 dataset. We achieve state-of-the-art results.")
    doc.save("test_pipeline.pdf")
    return "test_pipeline.pdf"

try:
    logger.info("Step 1: Create PDF")
    pdf_path = create_test_pdf()
    
    logger.info("Step 2: Import Modules")
    from backend.pdf_extractor import extract_text_from_pdf
    from backend.entity_extractor import extract_research_entities

    from backend.openalex_client import enrich_paper
    from backend.faiss_index import add_paper_to_index
    # from backend.chroma_store import get_chroma_store
    
    logger.info("Step 3: Extract Text")
    doc = extract_text_from_pdf(pdf_path)
    print(f"Extracted title: {doc.title}")
    
    logger.info("Step 4: Extract Entities")
    # Need paper_id for canonical json
    class MockDoc:
        def __init__(self, **kwargs):
            for k,v in kwargs.items(): setattr(self, k, v)
            
    canonical = extract_research_entities(doc, "test_paper_001")
    print(f"Extracted entities: {canonical.architecture}")
    
    logger.info("Step 5: Enrich Paper")
    try:
        enriched = enrich_paper(canonical)
        print(f"Enriched: {enriched.openalex['grounding_status']}")
    except Exception as e:
        print(f"Enrichment warning (expected if network fails): {e}")

    logger.info("Step 6: Index Paper")
    # Need to mock FAISS index if not exists?
    # backend/paper_index.py should handle it.
    try:
        ids = add_paper_to_index(canonical)
        print(f"Indexed {len(ids)} vectors")
    except Exception as e:
        print(f"Indexing failed: {e}")
        raise e

    logger.info("SUCCESS")

except Exception as e:
    logger.error(f"FAILURE: {e}")
    import traceback
    traceback.print_exc()
