#!/usr/bin/env python3
"""
Comprehensive test script to verify routing improvements work correctly.
"""

import os
import sys

# Add the current directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from routing import SimpleAgriculturalAI

def test_comprehensive_routing():
    """Test comprehensive routing functionality."""
    print("🧪 Comprehensive Routing Test")
    print("=" * 50)
    
    # Initialize the AI system
    ai = SimpleAgriculturalAI()
    
    # Test cases
    test_cases = [
        # Price-related queries (should go to price_prediction_agent)
        ("price of corn", "price_prediction_agent"),
        ("price of maize", "price_prediction_agent"),
        ("wheat price forecast", "price_prediction_agent"),
        ("onion market prices", "price_prediction_agent"),
        ("tomato price prediction", "price_prediction_agent"),
        ("rice commodity prices", "price_prediction_agent"),
        
        # Weather-related queries (should go to rain_forecast_agent)
        ("rainfall forecast for mumbai", "rain_forecast_agent"),
        ("weather in delhi", "rain_forecast_agent"),
        ("rain prediction for bangalore", "rain_forecast_agent"),
        ("storm forecast", "rain_forecast_agent"),
        
        # Other queries (should go to appropriate agents)
        ("how to control aphids", "farming_agent"),
        ("soil management tips", "farming_agent"),
        ("show me tractor images", "image_agent"),
        ("farming tutorial videos", "youtube_agent"),
    ]
    
    results = []
    
    for query, expected_agent in test_cases:
        print(f"\n🔍 Testing: '{query}'")
        print(f"   Expected agent: {expected_agent}")
        
        try:
            result = ai.ask(query)
            actual_agent = result.get('agent_used', 'unknown')
            success = result.get('success', False)
            
            print(f"   Actual agent: {actual_agent}")
            print(f"   Success: {success}")
            
            # Check if routing was correct
            if actual_agent == expected_agent:
                print("   ✅ Routing correct!")
                results.append(True)
            else:
                print(f"   ❌ Routing incorrect! Expected {expected_agent}, got {actual_agent}")
                results.append(False)
            
            # Show a preview of the response
            answer = result.get('answer', '')
            if answer:
                preview = answer[:100] + "..." if len(answer) > 100 else answer
                print(f"   Response preview: {preview}")
            
        except Exception as e:
            print(f"   ❌ Error: {str(e)}")
            results.append(False)
    
    # Summary
    print(f"\n📊 Test Summary")
    print("=" * 30)
    correct_routes = sum(results)
    total_tests = len(results)
    success_rate = (correct_routes / total_tests) * 100
    
    print(f"Total tests: {total_tests}")
    print(f"Correct routes: {correct_routes}")
    print(f"Success rate: {success_rate:.1f}%")
    
    if success_rate >= 80:
        print("🎉 Excellent routing performance!")
    elif success_rate >= 60:
        print("✅ Good routing performance!")
    else:
        print("⚠️ Routing needs improvement.")

if __name__ == "__main__":
    test_comprehensive_routing()
    print("\n✅ Comprehensive routing test completed!")

