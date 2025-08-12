#!/usr/bin/env python3
"""
Diagnostic test script for modular system issues
"""

import sys
import traceback

def test_imports():
    """Test all imports work correctly"""
    print("🔍 Testing imports...")
    
    try:
        from utils import AgentResult, Query, GROQ_API_KEY, get_llm_response, is_farming_query
        print("✅ utils imports successful")
    except Exception as e:
        print(f"❌ utils import failed: {e}")
        traceback.print_exc()
        return False
    
    try:
        from agents import BaseAgent, RishikhetAgent1, RishikhetAgent2, WebBasedAgent, FarmingAgent
        print("✅ agents imports successful")
    except Exception as e:
        print(f"❌ agents import failed: {e}")
        traceback.print_exc()
        return False
    
    try:
        from routing import LLMRouter, SimpleAgriculturalAI
        print("✅ routing imports successful")
    except Exception as e:
        print(f"❌ routing import failed: {e}")
        traceback.print_exc()
        return False
    
    return True

def test_llm_router():
    """Test LLM Router functionality"""
    print("\n🔍 Testing LLM Router...")
    
    try:
        from routing import LLMRouter, SimpleAgriculturalAI
        from agents import BaseAgent, FarmingAgent
        
        # Create a simple test setup
        agents = [FarmingAgent()]
        router = LLMRouter(agents)
        
        # Test city extraction
        city = router.extract_city_from_query("What's the weather in Mumbai?")
        print(f"✅ City extraction test: '{city}'")
        
        # Test routing
        agent_name = router.route_query("What fertilizer should I use for tomatoes?")
        print(f"✅ Routing test: '{agent_name}'")
        
        return True
        
    except Exception as e:
        print(f"❌ LLM Router test failed: {e}")
        traceback.print_exc()
        return False

def test_chain_of_thought():
    """Test chain of thought functionality"""
    print("\n🔍 Testing Chain of Thought...")
    
    try:
        from routing import SimpleAgriculturalAI
        
        ai = SimpleAgriculturalAI()
        result = ai.ask("What is the best fertilizer for tomatoes?")
        
        if 'chain_of_thought' in result:
            print(f"✅ Chain of thought present: {len(result['chain_of_thought'])} chars")
            print(f"Preview: {result['chain_of_thought'][:100]}...")
        else:
            print("❌ Chain of thought missing from result")
            print(f"Result keys: {list(result.keys())}")
        
        return 'chain_of_thought' in result
        
    except Exception as e:
        print(f"❌ Chain of thought test failed: {e}")
        traceback.print_exc()
        return False

def test_full_system():
    """Test the complete system"""
    print("\n🔍 Testing Full System...")
    
    try:
        from routing import SimpleAgriculturalAI
        
        ai = SimpleAgriculturalAI()
        
        # Test farming query
        result = ai.ask("How do I control pests in my tomato garden?")
        
        print(f"✅ System response received")
        print(f"Answer length: {len(result.get('answer', ''))}")
        print(f"Sources count: {len(result.get('sources', []))}")
        print(f"Chain of thought: {'✅' if result.get('chain_of_thought') else '❌'}")
        
        return True
        
    except Exception as e:
        print(f"❌ Full system test failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all diagnostic tests"""
    print("🚀 Starting Modular System Diagnostics...\n")
    
    tests = [
        ("Imports", test_imports),
        ("LLM Router", test_llm_router),
        ("Chain of Thought", test_chain_of_thought),
        ("Full System", test_full_system)
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results[test_name] = False
    
    print("\n" + "="*50)
    print("📊 DIAGNOSTIC RESULTS:")
    print("="*50)
    
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name:20} {status}")
    
    failed_tests = [name for name, passed in results.items() if not passed]
    if failed_tests:
        print(f"\n🔧 ISSUES FOUND: {', '.join(failed_tests)}")
        print("These components need fixing!")
    else:
        print("\n🎉 ALL TESTS PASSED! Modular system is working correctly.")

if __name__ == "__main__":
    main()
