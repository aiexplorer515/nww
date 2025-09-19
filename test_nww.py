"""
Simple test script for NWW package
"""

import sys
import os

# Add the package to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'nwwpkg'))

def test_imports():
    """Test that all modules can be imported."""
    try:
        from nwwpkg import ingest, preprocess, analyze, rules, score, fusion, eds, scenario, decider, ledger, eventblock, ui
        print("✅ All modules imported successfully!")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_basic_functionality():
    """Test basic functionality of key modules."""
    try:
        from nwwpkg.ingest import Extractor
        from nwwpkg.preprocess import Normalizer
        from nwwpkg.analyze import Tagger
        from nwwpkg.rules import Gating
        from nwwpkg.score import ScoreIS, ScoreDBN, LLMJudge
        
        # Test initialization
        extractor = Extractor("data/bundles/sample")
        normalizer = Normalizer()
        tagger = Tagger()
        gating = Gating()
        score_is = ScoreIS()
        score_dbn = ScoreDBN()
        score_llm = LLMJudge()
        
        print("✅ All modules initialized successfully!")
        return True
    except Exception as e:
        print(f"❌ Initialization error: {e}")
        return False

def main():
    """Run tests."""
    print("🧪 Testing NWW Package...")
    print("=" * 50)
    
    # Test imports
    print("Testing imports...")
    import_success = test_imports()
    
    # Test basic functionality
    print("\nTesting basic functionality...")
    functionality_success = test_basic_functionality()
    
    # Summary
    print("\n" + "=" * 50)
    if import_success and functionality_success:
        print("🎉 All tests passed! NWW package is ready to use.")
        return 0
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())



