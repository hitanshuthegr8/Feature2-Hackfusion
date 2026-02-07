import requests
import json
from typing import Dict, Optional
from config import Config
from utils.logging import setup_logger
from utils.retry import retry_with_backoff

logger = setup_logger(__name__)

class OllamaService:
    """Service for interacting with local Ollama instance."""
    
    def __init__(self):
        """Initialize Ollama service."""
        self.base_url = Config.OLLAMA_URL
        self.model = Config.OLLAMA_MODEL
        self.timeout = Config.OLLAMA_TIMEOUT
        
        # Strict extraction prompt
        self.extraction_prompt = """You are an information extraction engine.
Extract ONLY factual information explicitly present in the text.
If information is missing, return an empty list.

Output STRICT JSON with keys:
models, datasets, baselines

NO explanations.
NO guesses.
NO markdown.

Example output format:
{
  "models": ["ResNet-50", "Transformer"],
  "datasets": ["ImageNet", "CIFAR-10"],
  "baselines": [
    {"metric": "Accuracy", "value": "92.3%"},
    {"metric": "F1-score", "value": "0.88"}
  ]
}

Extract from the following text:"""
    
    @retry_with_backoff(max_attempts=2, backoff_factor=2)
    def generate(self, prompt: str, system_prompt: str = None) -> Optional[str]:
        """
        Generate text using Ollama.
        
        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            
        Returns:
            Generated text or None if generation failed
        """
        logger.info(f"Calling Ollama model: {self.model}")
        
        try:
            url = f"{self.base_url}/api/generate"
            
            payload = {
                'model': self.model,
                'prompt': prompt,
                'stream': False,
                'options': {
                    'temperature': 0.1,  # Low temperature for deterministic output
                    'num_predict': 1024
                }
            }
            
            if system_prompt:
                payload['system'] = system_prompt
            
            response = requests.post(
                url,
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            
            data = response.json()
            generated_text = data.get('response', '').strip()
            
            logger.info(f"Ollama generation completed ({len(generated_text)} chars)")
            return generated_text
        
        except requests.RequestException as e:
            logger.error(f"Ollama API request failed: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error in Ollama generation: {e}")
            return None
    
    def extract_structured_info(self, text: str) -> Dict:
        """
        Extract structured information from text using Ollama.
        
        Args:
            text: Text to extract information from
            
        Returns:
            Dictionary with extracted models, datasets, and baselines
        """
        logger.info(f"[DEBUG] Starting extraction on text of length: {len(text)} chars")
        
        # Truncate text if too long (keep beginning and end)
        max_chars = 6000  # Leave room for prompt
        if len(text) > max_chars:
            logger.warning(f"[DEBUG] Text too long ({len(text)} chars), truncating to {max_chars}")
            half = max_chars // 2
            text = text[:half] + "\n\n[...truncated...]\n\n" + text[-half:]
        
        # Build prompt
        full_prompt = f"{self.extraction_prompt}\n\n{text}"
        logger.info(f"[DEBUG] Prompt length: {len(full_prompt)} chars")
        logger.info(f"[DEBUG] Prompt preview (first 200 chars): {full_prompt[:200]}...")
        
        # Generate response
        logger.info("[DEBUG] Calling Ollama generate...")
        response = self.generate(full_prompt)
        
        if not response:
            logger.error("[DEBUG] ❌ Ollama returned empty response!")
            return self._empty_extraction()
        
        logger.info(f"[DEBUG] ✅ Ollama response received: {len(response)} chars")
        logger.info(f"[DEBUG] Response preview (first 500 chars):\n{response[:500]}")
        
        # Parse JSON response
        try:
            # Clean response (remove markdown code blocks if present)
            cleaned = response.strip()
            logger.info(f"[DEBUG] Cleaning response... starts with: {cleaned[:50]}")
            
            if cleaned.startswith('```'):
                logger.info("[DEBUG] Detected markdown code block, extracting JSON...")
                # Extract JSON from code block
                lines = cleaned.split('\n')
                cleaned = '\n'.join(lines[1:-1])
                logger.info(f"[DEBUG] Cleaned JSON (first 300 chars): {cleaned[:300]}")
            
            # Parse JSON
            logger.info("[DEBUG] Parsing JSON...")
            extracted = json.loads(cleaned)
            logger.info(f"[DEBUG] ✅ JSON parsed successfully: {type(extracted)}")
            
            # Validate structure
            if not isinstance(extracted, dict):
                logger.error(f"[DEBUG] ❌ Ollama response is not a dictionary, got: {type(extracted)}")
                return self._empty_extraction()
            
            logger.info(f"[DEBUG] Extracted keys: {list(extracted.keys())}")
            
            # Ensure all required keys exist
            result = {
                'models': extracted.get('models', []),
                'datasets': extracted.get('datasets', []),
                'baselines': extracted.get('baselines', [])
            }
            
            # Validate types
            if not isinstance(result['models'], list):
                result['models'] = []
            if not isinstance(result['datasets'], list):
                result['datasets'] = []
            if not isinstance(result['baselines'], list):
                result['baselines'] = []
            
            # Validate baselines structure
            validated_baselines = []
            for baseline in result['baselines']:
                if isinstance(baseline, dict) and 'metric' in baseline and 'value' in baseline:
                    validated_baselines.append(baseline)
            result['baselines'] = validated_baselines
            
            logger.info(f"[DEBUG] ✅ Extraction complete:")
            logger.info(f"[DEBUG]   - Models: {len(result['models'])} found -> {result['models']}")
            logger.info(f"[DEBUG]   - Datasets: {len(result['datasets'])} found -> {result['datasets']}")
            logger.info(f"[DEBUG]   - Baselines: {len(result['baselines'])} found -> {result['baselines']}")
            
            logger.info(
                f"Extracted: {len(result['models'])} models, "
                f"{len(result['datasets'])} datasets, "
                f"{len(result['baselines'])} baselines"
            )
            
            return result
        
        
        except json.JSONDecodeError as e:
            logger.error(f"[DEBUG] ❌ Failed to parse Ollama JSON response: {e}")
            logger.error(f"[DEBUG] Error at position: line {e.lineno}, column {e.colno}")
            logger.error(f"[DEBUG] Full raw response:\n{response}")
            return self._empty_extraction()
        except Exception as e:
            logger.error(f"[DEBUG] ❌ Unexpected error parsing Ollama response: {e}")
            logger.error(f"[DEBUG] Exception type: {type(e).__name__}")
            import traceback
            logger.error(f"[DEBUG] Traceback:\n{traceback.format_exc()}")
            return self._empty_extraction()
    
    def _empty_extraction(self) -> Dict:
        """Return empty extraction result."""
        return {
            'models': [],
            'datasets': [],
            'baselines': []
        }
