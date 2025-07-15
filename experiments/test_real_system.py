# experiments/test_real_system.py
"""
🧪 TEST YOUR REAL SYSTEM - Fixed version
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_your_real_system():
    """Test your actual EpisodicMemoryLLM_FINAL system"""
    
    print("🧪 TESTING YOUR REAL SYSTEM")
    print("=" * 50)
    
    try:
        # Import your system
        from src.models.episodic_memory_llm import EpisodicMemoryLLM_FINAL
        
        print("✅ Importing your system...")
        system = EpisodicMemoryLLM_FINAL(
            model_name="gpt2-medium",
            device="cpu",
            tkg_max_nodes=1000
        )
        print("✅ System loaded successfully!")
        
        # Test with simple conversation
        print("\n📝 Testing basic conversation:")
        
        # Add some context
        context = [
            "Hi, I'm Dr. Maria Santos, a professor of biology at UC Berkeley.",
            "I completed my PhD in Marine Biology from Scripps Institution in 2012.",
            "In my free time, I enjoy scuba diving and underwater photography."
        ]
        
        for ctx in context:
            result = system.chat_breakthrough(ctx)
            print(f"  Added: {ctx[:50]}...")
        
        # Test queries
        queries = [
            "What is your name and profession?",
            "Where did you get your PhD and when?",
            "What are your research interests and hobbies?"
        ]
        
        print(f"\n🔍 Testing queries:")
        for query in queries:
            result = system.chat_breakthrough(query)
            response = result["response"]
            strategy = result["performance"]["strategy"]
            confidence = result["performance"]["confidence"]
            
            print(f"  Q: {query}")
            print(f"  A: {response}")
            print(f"  Strategy: {strategy}, Confidence: {confidence:.2f}")
            print()
        
        # Get system stats
        stats = system.get_research_statistics()
        print(f"📊 System Stats:")
        print(f"  Memory Accuracy: {stats['memory_accuracy']:.1%}")
        print(f"  TKG Nodes: {stats['tkg_nodes']}")
        print(f"  Avg Response Time: {stats['avg_response_time_ms']:.1f}ms")
        
        return system
        
    except ImportError as e:
        print(f"❌ Could not import your system: {e}")
        print("Make sure the file path is correct")
        return None
    except Exception as e:
        print(f"❌ Error testing your system: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_extraction_integration():
    """Test if generic extractor integrates with your system"""
    
    print("\n🔧 TESTING EXTRACTION INTEGRATION")
    print("=" * 50)
    
    try:
        from src.extraction.generic_information_extractor import GenericInformationExtractor
        
        extractor = GenericInformationExtractor()
        
        test_text = "Hi, I'm Dr. Elena Rodriguez and I work as a research scientist at MIT."
        
        print(f"Testing: {test_text}")
        
        results = extractor.extract_comprehensive_information(test_text)
        
        print(f"📊 Extraction successful!")
        print(f"  Personal info confidence: {results['personal_info']['confidence']:.2f}")
        print(f"  Professional info confidence: {results['professional_info']['confidence']:.2f}")
        
        # Check if we can get names
        if results['personal_info']['names']:
            name_data = results['personal_info']['names'][0]
            print(f"  Extracted name: {name_data['text']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Extraction integration failed: {e}")
        return False

if __name__ == "__main__":
    # Test your real system
    system = test_your_real_system()
    
    # Test extraction integration
    extraction_works = test_extraction_integration()
    
    if system and extraction_works:
        print(f"\n✅ READY FOR INTEGRATION!")
        print(f"Next step: Replace hardcoded extraction in your TKG")
    else:
        print(f"\n❌ NEED TO FIX IMPORTS FIRST")
        print(f"Check file paths and dependencies")