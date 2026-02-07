"""
Quick test to verify the backend server is running and accessible
"""
import requests
import sys

API_BASE = "http://localhost:5000"

def test_server():
    print("Testing backend server connection...")
    print(f"API URL: {API_BASE}\n")
    
    try:
        # Test health endpoint
        print("1. Testing /health endpoint...")
        response = requests.get(f"{API_BASE}/health", timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            print("   ✅ Server is running!")
            print(f"   - Status: {data.get('status', 'unknown')}")
            print(f"   - Indexed papers: {data.get('indexed_papers', 0)}")
            print(f"   - ChromaDB papers: {data.get('chroma_papers', 0)}")
            print(f"   - Ollama status: {data.get('ollama_status', 'unknown')}")
            return True
        else:
            print(f"   ❌ Server returned status {response.status_code}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("   ❌ Cannot connect to server!")
        print("\n   Make sure the server is running:")
        print("   python run_server.py")
        return False
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

if __name__ == "__main__":
    success = test_server()
    sys.exit(0 if success else 1)

