"""
Simple demonstration of LangGraph integration in Agricultural AI System.
"""

import os
import sys

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def simple_demo():
    """Simple demo to show LangGraph integration."""
    print("🌾 LangGraph Agricultural AI Demo")
    print("=" * 40)
    
    try:
        # Import the enhanced system
        from routing import EnhancedAgriculturalAI, LANGGRAPH_AVAILABLE
        
        print(f"LangGraph Available: {LANGGRAPH_AVAILABLE}")
        
        if LANGGRAPH_AVAILABLE:
            print("\n🚀 Testing LangGraph System...")
            ai = EnhancedAgriculturalAI(use_langgraph=True)
        else:
            print("\n📊 Using Traditional System (LangGraph not available)...")
            ai = EnhancedAgriculturalAI(use_langgraph=False)
        
        # Show system info
        info = ai.get_system_info()
        print(f"Current System: {info['system_type']}")
        
        # Test queries
        test_queries = [
            "How do I improve soil fertility?",
            "What are common tomato diseases?"
        ]
        
        for query in test_queries:
            print(f"\n🔍 Query: {query}")
            print("🤖 Processing...")
            
            try:
                result = ai.ask(query)
                print(f"✅ Agent: {result['agent_name']}")
                print(f"✅ System: {result.get('system_type', 'unknown')}")
                print(f"✅ Confidence: {result['confidence']:.2f}")
                print(f"✅ Response: {result['response'][:150]}...")
                
                if result.get('sources'):
                    print(f"✅ Sources: {len(result['sources'])} found")
                    
            except Exception as e:
                print(f"❌ Error: {str(e)}")
        
        print(f"\n🎉 Demo completed successfully!")
        
    except Exception as e:
        print(f"❌ Demo failed: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    simple_demo()
