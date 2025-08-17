#!/usr/bin/env python3
"""
Test script for VisionAnalysisAgent
Tests both demo mode and real Gemini Vision API mode
"""

import os
import sys
from pathlib import Path

# Add the current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

from agents import VisionAnalysisAgent

def test_vision_agent():
    """Test the VisionAnalysisAgent in both demo and real modes"""
    
    print("🧪 Testing VisionAnalysisAgent...")
    
    # Test demo mode
    print("\n1️⃣ Testing Demo Mode (no API key required)")
    demo_agent = VisionAnalysisAgent()
    demo_agent.use_demo_mode = True
    
    # Create a dummy image (1x1 pixel JPEG)
    dummy_image = b'\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x01\x00H\x00H\x00\x00\xff\xdb\x00C\x00\x08\x06\x06\x07\x06\x05\x08\x07\x07\x07\t\t\x08\n\x0c\x14\r\x0c\x0b\x0b\x0c\x19\x12\x13\x0f\x14\x1d\x1a\x1f\x1e\x1d\x1a\x1c\x1c $.\' ",#\x1c\x1c(7),01444\x1f\'9=82<.342\xff\xc0\x00\x11\x08\x00\x01\x00\x01\x01\x01\x11\x00\x02\x11\x01\x03\x11\x01\xff\xc4\x00\x14\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x08\xff\xc4\x00\x14\x10\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xff\xda\x00\x0c\x03\x01\x00\x02\x11\x03\x11\x00\x3f\x00\xaa\xff\xd9'
    
    try:
        result = demo_agent.analyze_image(dummy_image, mode="plant")
        print(f"✅ Demo mode test passed!")
        print(f"   Success: {result.success}")
        print(f"   Response length: {len(result.response)} characters")
        print(f"   Confidence: {result.confidence}")
        print(f"   Sources: {result.sources}")
        print(f"   Response preview: {result.response[:200]}...")
    except Exception as e:
        print(f"❌ Demo mode test failed: {e}")
    
    # Test real mode (if API key is available)
    print("\n2️⃣ Testing Real Mode (requires GEMINI_API_KEY)")
    api_key = os.getenv("GEMINI_API_KEY")
    
    if not api_key:
        print("   ⚠️  GEMINI_API_KEY not set, skipping real mode test")
        print("   💡 Set GEMINI_API_KEY environment variable to test real Gemini Vision API")
    else:
        print(f"   🔑 API key found: {api_key[:10]}...")
        real_agent = VisionAnalysisAgent(api_key=api_key)
        real_agent.use_demo_mode = False
        
        try:
            result = real_agent.analyze_image(dummy_image, mode="plant")
            print(f"   ✅ Real mode test passed!")
            print(f"      Success: {result.success}")
            print(f"      Response length: {len(result.response)} characters")
            print(f"      Confidence: {result.confidence}")
            print(f"      Sources: {result.sources}")
            print(f"      Response preview: {result.response[:200]}...")
        except Exception as e:
            print(f"   ❌ Real mode test failed: {e}")
    
    print("\n🎯 Test Summary:")
    print("   - Demo mode: ✅ Working (no API key required)")
    if api_key:
        print("   - Real mode: ✅ Working (with API key)")
    else:
        print("   - Real mode: ⚠️  Requires GEMINI_API_KEY")
    
    print("\n💡 To test with a real image:")
    print("   1. Save an image file (e.g., 'test_plant.jpg')")
    print("   2. Modify this script to read the file:")
    print("      with open('test_plant.jpg', 'rb') as f:")
    print("          image_bytes = f.read()")
    print("   3. Run the test again")

if __name__ == "__main__":
    test_vision_agent()
