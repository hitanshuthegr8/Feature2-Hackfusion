from typing import List, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from config import Config
from utils.logging import setup_logger

logger = setup_logger(__name__)

class TFIDFService:
    """Service for extracting weighted keywords using TF-IDF."""
    
    def __init__(self):
        """Initialize TF-IDF service with custom stopwords."""
        # Common academic stopwords to exclude
        self.stopwords = set([
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
            'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
            'should', 'could', 'may', 'might', 'must', 'can', 'this', 'that',
            'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they',
            'what', 'which', 'who', 'when', 'where', 'why', 'how', 'paper',
            'study', 'research', 'using', 'based', 'approach', 'method'
        ])
    
    def extract_keywords(
        self,
        query: str,
        max_features: int = None
    ) -> List[Tuple[str, float]]:
        """
        Extract weighted keywords from query using TF-IDF.
        
        Args:
            query: Research query string
            max_features: Maximum number of keywords to extract
            
        Returns:
            List of (keyword, weight) tuples sorted by weight descending
        """
        if not query or not query.strip():
            logger.warning("Empty query provided to TF-IDF service")
            return []
        
        max_features = max_features or Config.TFIDF_MAX_FEATURES
        
        try:
            # Create vectorizer with custom settings
            vectorizer = TfidfVectorizer(
                max_features=max_features,
                stop_words=list(self.stopwords),
                ngram_range=Config.TFIDF_NGRAM_RANGE,
                lowercase=True,
                token_pattern=r'\b[a-zA-Z][a-zA-Z0-9\-]*\b'  # Allow hyphens
            )
            
            # Fit and transform
            tfidf_matrix = vectorizer.fit_transform([query])
            feature_names = vectorizer.get_feature_names_out()
            
            # Extract weights
            weights = tfidf_matrix.toarray()[0]
            
            # Create keyword-weight pairs
            keywords = [
                (feature_names[i], float(weights[i]))
                for i in range(len(feature_names))
                if weights[i] > 0
            ]
            
            # Sort by weight descending
            keywords.sort(key=lambda x: x[1], reverse=True)
            
            logger.info(
                f"Extracted {len(keywords)} keywords from query: {query[:100]}..."
            )
            
            return keywords
        
        except Exception as e:
            logger.error(f"TF-IDF extraction failed: {e}")
            # Fallback: simple word extraction
            words = query.lower().split()
            unique_words = [
                w for w in words
                if w not in self.stopwords and len(w) > 2
            ][:max_features]
            return [(w, 1.0 / len(unique_words)) for w in unique_words]
