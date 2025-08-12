#!/usr/bin/env python3
"""
Test script to verify and fix cascading agent logic
"""

import sys
import os

def test_cascading_logic():
    """Test the cascading agent logic"""
    print("🔍 Testing Cascading Agent Logic...")
    
    try:
        # Import the system
        from routing import SimpleAgriculturalAI
        from utils import is_farming_query
        
        # Create AI instance
        ai = SimpleAgriculturalAI()
        
        # Test farming query detection
        test_question = "What fertilizer should I use for tomatoes?"
        is_farming = is_farming_query(test_question)
        print(f"✅ Farming query detection: '{test_question}' -> {is_farming}")
        
        # Test the full ask method
        print(f"\n🚀 Testing full ask method with: '{test_question}'")
        result = ai.ask(test_question)
        
        print("\n📊 RESULTS:")
        print(f"Answer length: {len(result.get('answer', ''))}")
        print(f"Sources count: {len(result.get('sources', []))}")
        print(f"Agent used: {result.get('agent_used', 'Unknown')}")
        print(f"Success: {result.get('success', False)}")
        
        # Check chain of thought for cascading evidence
        chain_of_thought = result.get('chain_of_thought', '')
        print(f"\n🔗 Chain of Thought Analysis:")
        print(f"Chain length: {len(chain_of_thought)}")
        
        # Look for evidence of cascading
        cascading_indicators = [
            "cascading agent",
            "Step 1: Trying Rishikhet",
            "Step 2: Getting general farming",
            "Step 3: Getting current web",
            "Specialized Knowledge",
            "General Farming Knowledge",
            "Current Web Information"
        ]
        
        found_indicators = []
        for indicator in cascading_indicators:
            if indicator.lower() in chain_of_thought.lower():
                found_indicators.append(indicator)
        
        print(f"Cascading indicators found: {len(found_indicators)}/{len(cascading_indicators)}")
        for indicator in found_indicators:
            print(f"  ✅ Found: {indicator}")
        
        missing_indicators = [ind for ind in cascading_indicators if ind not in found_indicators]
        for indicator in missing_indicators:
            print(f"  ❌ Missing: {indicator}")
        
        # Show partial chain of thought
        if chain_of_thought:
            print(f"\n📝 Chain of Thought Preview:")
            print(chain_of_thought[:500] + "..." if len(chain_of_thought) > 500 else chain_of_thought)
        
        # Determine if cascading is working
        is_cascading_working = len(found_indicators) >= 3
        print(f"\n🎯 CASCADING STATUS: {'✅ WORKING' if is_cascading_working else '❌ BROKEN'}")
        
        return is_cascading_working, result
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None

def diagnose_issue():
    """Diagnose why cascading might not be working"""
    print("\n🔍 Diagnosing Cascading Issues...")
    
    try:
        from routing import SimpleAgriculturalAI
        from utils import is_farming_query
        
        ai = SimpleAgriculturalAI()
        
        # Check agent initialization
        print(f"Rishikhet agents: {len(ai.rishikhet_agents)}")
        print(f"Farming agent: {'✅' if ai.farming_agent else '❌'}")
        print(f"Web agent: {'✅' if ai.web_based_agent else '❌'}")
        print(f"Router: {'✅' if ai.router else '❌'}")
        
        # Test farming query detection
        test_queries = [
            "What fertilizer should I use for tomatoes?",
            "How to control pests in my garden?",
            "Best practices for organic farming?",
            "What's the weather in Mumbai?"  # Non-farming
        ]
        
        for query in test_queries:
            is_farming = is_farming_query(query)
            print(f"Query: '{query[:30]}...' -> Farming: {is_farming}")
        
        return True
        
    except Exception as e:
        print(f"❌ Diagnosis failed: {e}")
        return False

def main():
    """Main test function"""
    print("🚀 Cascading Agent Logic Test & Fix")
    print("=" * 50)
    
    # Run diagnosis
    diagnosis_success = diagnose_issue()
    
    if diagnosis_success:
        # Run cascading test
        cascading_working, result = test_cascading_logic()
        
        if not cascading_working:
            print("\n🔧 ISSUE IDENTIFIED:")
            print("Cascading logic is not working properly!")
            print("The system is likely only using the first successful agent instead of all agents.")
            
            print("\n💡 RECOMMENDED FIX:")
            print("1. Ensure all agents (Rishikhet, Farming, WebBased) are being called")
            print("2. Verify that responses from all agents are being collected")
            print("3. Check that synthesis is combining all responses")
            
        else:
            print("\n🎉 Cascading logic is working correctly!")
    
    else:
        print("\n❌ Could not complete diagnosis due to import/initialization errors")

if __name__ == "__main__":
    main()
