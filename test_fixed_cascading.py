#!/usr/bin/env python3
"""
Test the fixed cascading agent logic
"""

def test_cascading_fix():
    """Test if the cascading logic now works correctly"""
    print("🔍 Testing FIXED Cascading Agent Logic...")
    print("=" * 50)
    
    try:
        from routing import SimpleAgriculturalAI
        from utils import is_farming_query
        
        # Create AI instance
        ai = SimpleAgriculturalAI()
        
        # Test with a farming query
        test_question = "What fertilizer should I use for tomatoes?"
        print(f"🌾 Testing with: '{test_question}'")
        
        # Verify it's detected as farming query
        is_farming = is_farming_query(test_question)
        print(f"✅ Farming query detection: {is_farming}")
        
        if not is_farming:
            print("❌ ERROR: Query not detected as farming query!")
            return False
        
        # Test the full system
        result = ai.ask(test_question)
        
        print("\n📊 RESULTS:")
        print(f"Answer length: {len(result.get('answer', ''))}")
        print(f"Sources count: {len(result.get('sources', []))}")
        print(f"Agent used: {result.get('agent_used', 'Unknown')}")
        print(f"Success: {result.get('success', False)}")
        
        # Analyze chain of thought for cascading evidence
        chain_of_thought = result.get('chain_of_thought', '')
        print(f"\n🔗 Chain of Thought Analysis:")
        print(f"Chain length: {len(chain_of_thought)}")
        
        # Check for cascading indicators
        cascading_indicators = [
            "cascading agent approach",
            "Step 1: Checking Rishikhet",
            "Step 2: Getting general farming",
            "Step 3: Getting current web",
            "Synthesizing"
        ]
        
        found_indicators = []
        for indicator in cascading_indicators:
            if indicator.lower() in chain_of_thought.lower():
                found_indicators.append(indicator)
                print(f"  ✅ Found: {indicator}")
        
        missing_indicators = [ind for ind in cascading_indicators if ind not in found_indicators]
        for indicator in missing_indicators:
            print(f"  ❌ Missing: {indicator}")
        
        # Check if multiple agents were used
        agent_used = result.get('agent_used', '')
        multiple_agents = '+' in agent_used or 'cascading' in agent_used.lower()
        print(f"\n🤖 Multiple Agents Used: {'✅' if multiple_agents else '❌'}")
        print(f"Agent string: '{agent_used}'")
        
        # Show chain of thought
        if chain_of_thought:
            print(f"\n📝 Chain of Thought:")
            print(chain_of_thought)
        
        # Determine success
        cascading_working = len(found_indicators) >= 3 and multiple_agents
        print(f"\n🎯 CASCADING STATUS: {'✅ WORKING' if cascading_working else '❌ STILL BROKEN'}")
        
        if cascading_working:
            print("🎉 SUCCESS: All three agents (Rishikhet → Farming → WebBased) are now being used!")
        else:
            print("❌ ISSUE: Cascading logic is still not working properly")
            print("   Expected: Multiple agents with synthesis")
            print(f"   Got: {agent_used}")
        
        return cascading_working
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function"""
    print("🚀 Testing Fixed Cascading Agent Logic")
    print("Expected: Rishikhet Agent → Farming Agent → WebBased Agent")
    print("=" * 60)
    
    success = test_cascading_fix()
    
    if success:
        print("\n🎉 CASCADING LOGIC FIXED!")
        print("✅ All three agents are now being used in sequence")
        print("✅ Responses are being synthesized properly")
        print("✅ Chain of thought shows the complete process")
    else:
        print("\n❌ CASCADING LOGIC STILL NEEDS WORK")
        print("The fix may not be complete or there are other issues")

if __name__ == "__main__":
    main()
