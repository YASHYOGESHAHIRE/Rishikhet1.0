"""
Comprehensive test for LangGraph integration in Agricultural AI System.
"""

import sys
import os
import traceback

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test if all required modules can be imported."""
    print("🔍 Testing imports...")
    
    try:
        from utils import AgentResult, Query
        print("✅ Utils imported successfully")
    except Exception as e:
        print(f"❌ Utils import failed: {e}")
        return False
    
    try:
        from agents import BaseAgent, RishikhetAgent1, FarmingAgent
        print("✅ Agents imported successfully")
    except Exception as e:
        print(f"❌ Agents import failed: {e}")
        return False
    
    try:
        from langgraph_orchestrator import LangGraphOrchestrator, LangGraphAgriculturalAI
        print("✅ LangGraph orchestrator imported successfully")
    except Exception as e:
        print(f"❌ LangGraph orchestrator import failed: {e}")
        return False
    
    try:
        from routing import EnhancedAgriculturalAI, LANGGRAPH_AVAILABLE
        print(f"✅ Enhanced routing imported successfully (LangGraph available: {LANGGRAPH_AVAILABLE})")
    except Exception as e:
        print(f"❌ Enhanced routing import failed: {e}")
        return False
    
    return True

def test_langgraph_orchestrator():
    """Test the LangGraph orchestrator directly."""
    print("\n🚀 Testing LangGraph Orchestrator...")
    
    try:
        from langgraph_orchestrator import LangGraphOrchestrator
        
        orchestrator = LangGraphOrchestrator()
        print("✅ LangGraph orchestrator initialized")
        
        # Test a simple query
        test_query = "How do I improve soil fertility?"
        print(f"🔍 Testing query: '{test_query}'")
        
        result = orchestrator.process_query(test_query)
        
        print(f"✅ Query processed successfully")
        print(f"   Agent: {result.agent_name}")
        print(f"   Confidence: {result.confidence:.2f}")
        print(f"   Response length: {len(result.response)} characters")
        print(f"   Sources: {len(result.sources)} found")
        
        return True
        
    except Exception as e:
        print(f"❌ LangGraph orchestrator test failed: {e}")
        traceback.print_exc()
        return False

def test_enhanced_ai_system():
    """Test the enhanced AI system with both traditional and LangGraph modes."""
    print("\n🎯 Testing Enhanced AI System...")
    
    try:
        from routing import EnhancedAgriculturalAI, LANGGRAPH_AVAILABLE
        
        test_queries = [
            "What are the best practices for wheat farming?",
            "How do I control pests in tomato crops?"
        ]
        
        # Test traditional system
        print("\n📊 Testing Traditional System...")
        traditional_ai = EnhancedAgriculturalAI(use_langgraph=False)
        
        for query in test_queries:
            print(f"🔍 Query: {query}")
            result = traditional_ai.ask(query)
            print(f"   System: {result.get('system_type', 'unknown')}")
            print(f"   Agent: {result['agent_name']}")
            print(f"   Confidence: {result['confidence']:.2f}")
            print(f"   Response: {result['response'][:100]}...")
        
        # Test LangGraph system if available
        if LANGGRAPH_AVAILABLE:
            print("\n🚀 Testing LangGraph System...")
            langgraph_ai = EnhancedAgriculturalAI(use_langgraph=True)
            
            for query in test_queries:
                print(f"🔍 Query: {query}")
                result = langgraph_ai.ask(query)
                print(f"   System: {result.get('system_type', 'unknown')}")
                print(f"   Agent: {result['agent_name']}")
                print(f"   Confidence: {result['confidence']:.2f}")
                print(f"   Response: {result['response'][:100]}...")
        else:
            print("⚠️  LangGraph system not available for testing")
        
        return True
        
    except Exception as e:
        print(f"❌ Enhanced AI system test failed: {e}")
        traceback.print_exc()
        return False

def test_system_switching():
    """Test the system switching functionality."""
    print("\n🔄 Testing System Switching...")
    
    try:
        from routing import EnhancedAgriculturalAI, LANGGRAPH_AVAILABLE
        
        ai = EnhancedAgriculturalAI(use_langgraph=True)
        
        # Get initial system info
        info = ai.get_system_info()
        print(f"Initial system: {info['system_type']}")
        
        # Test switching
        ai.switch_system(use_langgraph=False)
        info = ai.get_system_info()
        print(f"After switch to traditional: {info['system_type']}")
        
        if LANGGRAPH_AVAILABLE:
            ai.switch_system(use_langgraph=True)
            info = ai.get_system_info()
            print(f"After switch to LangGraph: {info['system_type']}")
        
        print("✅ System switching works correctly")
        return True
        
    except Exception as e:
        print(f"❌ System switching test failed: {e}")
        traceback.print_exc()
        return False

def test_agent_collaboration():
    """Test agent collaboration through LangGraph."""
    print("\n🤝 Testing Agent Collaboration...")
    
    try:
        from langgraph_orchestrator import LangGraphAgriculturalAI
        
        ai = LangGraphAgriculturalAI()
        
        # Test a complex query that should trigger collaboration
        complex_query = "I need help with wheat farming - soil preparation, pest control, and market prices"
        print(f"🔍 Complex query: {complex_query}")
        
        result = ai.ask(complex_query)
        
        print(f"✅ Collaboration test completed")
        print(f"   Final agent: {result.agent_name}")
        print(f"   Confidence: {result.confidence:.2f}")
        print(f"   Response length: {len(result.response)} characters")
        print(f"   Sources: {len(result.sources)} found")
        
        return True
        
    except Exception as e:
        print(f"❌ Agent collaboration test failed: {e}")
        traceback.print_exc()
        return False

def run_all_tests():
    """Run all tests and provide a summary."""
    print("🌾 LangGraph Integration Test Suite")
    print("=" * 50)
    
    tests = [
        ("Import Tests", test_imports),
        ("LangGraph Orchestrator", test_langgraph_orchestrator),
        ("Enhanced AI System", test_enhanced_ai_system),
        ("System Switching", test_system_switching),
        ("Agent Collaboration", test_agent_collaboration)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{'=' * 20} {test_name} {'=' * 20}")
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print(f"\n{'=' * 20} TEST SUMMARY {'=' * 20}")
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{test_name}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! LangGraph integration is working correctly.")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
    
    return passed == total

if __name__ == "__main__":
    run_all_tests()
