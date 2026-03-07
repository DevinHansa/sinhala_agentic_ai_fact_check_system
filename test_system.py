import os
import sys

# Add root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.workflow import FactCheckingWorkflow
from src.vector_store import QdrantVectorStore
from dotenv import load_dotenv

# Load env
load_dotenv()

def test_workflow():
    print("Initializing components...")
    # We might need to handle Qdrant lock if app is running.
    # For test, we can use in-memory or just try.
    try:
        vs = QdrantVectorStore()
        workflow = FactCheckingWorkflow(vs)
        
        statement = "ශ්‍රී ලංකාවේ ජනාධිපතිවරණය 2024 පැවැත්වේ." # "Sri Lanka Presidential Election is held in 2024"
        print(f"Testing Statement: {statement}")
        
        result = workflow.verify(statement)
        
        print("\n--- Result ---")
        print(f"Domain: {result.get('domain')}")
        print(f"Verdict: {result.get('verdict')}")
        print(f"Analysis Preview: {result.get('analysis')[:100]}...")
        
        # Verify structure
        assert "domain" in result
        assert "verdict" in result
        assert "analysis" in result
        assert "search_results" in result
        
        print("\n✅ Test Passed!")
        
    except Exception as e:
        print(f"\n❌ Test Failed: {e}")

if __name__ == "__main__":
    test_workflow()
