import time
from typing import List, Dict, Optional
import requests
from config import Config
from utils.logging import setup_logger
from utils.retry import retry_with_backoff
from utils.cache import Cache

logger = setup_logger(__name__)

class OpenAlexService:
    """Service for querying OpenAlex API."""
    
    def __init__(self):
        """Initialize OpenAlex service with caching."""
        self.base_url = Config.OPENALEX_API_URL
        self.email = Config.OPENALEX_EMAIL
        self.cache = Cache(Config.CACHE_DIR / "openalex")
    
    @retry_with_backoff(max_attempts=3, backoff_factor=2)
    def _make_request(self, url: str, params: Dict) -> Dict:
        """
        Make HTTP request to OpenAlex API with retry logic.
        
        Args:
            url: API endpoint URL
            params: Query parameters
            
        Returns:
            JSON response data
        """
        headers = {
            'User-Agent': f'ResearchServer/1.0 (mailto:{self.email})'
        }
        
        response = requests.get(
            url,
            params=params,
            headers=headers,
            timeout=30
        )
        response.raise_for_status()
        
        # Respect rate limits
        time.sleep(Config.OPENALEX_RATE_LIMIT_DELAY)
        
        return response.json()
    
    def search_papers(
        self,
        keywords: List[str],
        max_results: int = None
    ) -> List[Dict]:
        """
        Search for papers using keywords.
        
        Args:
            keywords: List of search keywords
            max_results: Maximum number of results to return
            
        Returns:
            List of paper metadata dictionaries
        """
        max_results = max_results or Config.OPENALEX_MAX_RESULTS
        
        # Build search query
        search_query = ' '.join(keywords)
        cache_key = f"search:{search_query}:{max_results}"
        
        # Check cache
        cached_results = self.cache.get(cache_key)
        if cached_results is not None:
            return cached_results
        
        logger.info(f"Searching OpenAlex for: {search_query}")
        
        try:
            url = f"{self.base_url}/works"
            params = {
                'search': search_query,
                'per-page': max_results,
                'filter': 'has_fulltext:true,is_oa:true',  # Only open access with PDFs
                'sort': 'relevance_score:desc'
            }
            
            data = self._make_request(url, params)
            
            # Extract and normalize results
            papers = []
            for work in data.get('results', []):
                paper = self._normalize_paper(work)
                if paper:
                    papers.append(paper)
            
            logger.info(f"Found {len(papers)} papers with PDFs")
            
            # Cache results
            self.cache.set(cache_key, papers)
            
            return papers
        
        except requests.RequestException as e:
            logger.error(f"OpenAlex API request failed: {e}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error in OpenAlex search: {e}")
            return []
    
    def _normalize_paper(self, work: Dict) -> Optional[Dict]:
        """
        Normalize OpenAlex work data into consistent format.
        
        Args:
            work: Raw work data from OpenAlex
            
        Returns:
            Normalized paper dictionary or None if invalid
        """
        try:
            # Find PDF location
            pdf_url = None
            locations = work.get('locations', [])
            
            for location in locations:
                if location.get('pdf_url'):
                    pdf_url = location['pdf_url']
                    break
            
            # Skip if no PDF available
            if not pdf_url:
                return None
            
            # Extract authors
            authors = []
            for authorship in work.get('authorships', [])[:5]:  # Limit to first 5
                author = authorship.get('author', {})
                if author.get('display_name'):
                    authors.append(author['display_name'])
            
            # Normalize paper data
            paper = {
                'id': work.get('id', '').split('/')[-1],  # Extract OpenAlex ID
                'title': work.get('title', 'Unknown Title'),
                'authors': authors,
                'year': work.get('publication_year'),
                'doi': work.get('doi'),
                'pdf_url': pdf_url,
                'abstract': work.get('abstract'),
                'cited_by_count': work.get('cited_by_count', 0),
                'venue': self._extract_venue(work),
                'concepts': self._extract_concepts(work)
            }
            
            return paper
        
        except Exception as e:
            logger.warning(f"Failed to normalize paper: {e}")
            return None
    
    def _extract_venue(self, work: Dict) -> Optional[str]:
        """Extract publication venue from work data."""
        location = work.get('primary_location', {})
        source = location.get('source')
        if source:
            return source.get('display_name')
        return None
    
    def _extract_concepts(self, work: Dict) -> List[str]:
        """Extract top concepts from work data."""
        concepts = work.get('concepts', [])
        return [
            c['display_name']
            for c in concepts[:5]  # Top 5 concepts
            if c.get('score', 0) > 0.3  # Only significant concepts
        ]
