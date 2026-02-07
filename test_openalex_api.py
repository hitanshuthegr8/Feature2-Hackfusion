"""
Test script to verify OpenAlex API calls are working correctly
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backend.openalex_client import search_by_topic
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def test_openalex_search():
    """Test OpenAlex search with various queries"""
    
    test_queries = [
        "vision transformer",
        "medical image segmentation",
        "deep learning",
        "neural networks"
    ]
    
    print("=" * 60)
    print("Testing OpenAlex API Calls")
    print("=" * 60)
    
    for query in test_queries:
        print(f"\n🔍 Testing query: '{query}'")
        print("-" * 60)
        
        try:
            results = search_by_topic(query, limit=5)
            
            if results:
                print(f"✅ SUCCESS: Found {len(results)} papers")
                print(f"   First paper: {results[0].get('title', 'Unknown')[:60]}...")
                print(f"   OpenAlex ID: {results[0].get('openalex_id', 'N/A')}")
                print(f"   Citations: {results[0].get('cited_by_count', 0)}")
            else:
                print(f"❌ FAILED: No papers found")
                print(f"   This might indicate:")
                print(f"   - Rate limit hit (wait 5-10 minutes)")
                print(f"   - Query too specific")
                print(f"   - API issue")
                
        except Exception as e:
            print(f"❌ ERROR: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("Test Complete")
    print("=" * 60)

if __name__ == "__main__":
    test_openalex_search()

