
"""
Test script to verify PricePredictionAgent works with user profiles.
"""

import os
import sys
import json
from datetime import datetime

# Add the current directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agents import PricePredictionAgent
from utils import Query
from profile_models import FarmerProfile, ProfileManager

def create_test_profile():
    """Create a test farmer profile."""
    profile = FarmerProfile(
        farmer_id="test_farmer_001",
        name="Test Farmer",
        email="test@example.com",
        phone="1234567890",
        location="Dhule, Maharashtra, India",
        latitude=20.9024,
        longitude=74.7774,
        farm_size_acres=50.0,
        primary_crops=["Onion", "Tomato", "Wheat"],
        secondary_crops=["Corn", "Soybeans"],
        livestock=[],
        farming_type="conventional",
        experience_years=15,
        soil_type="loamy",
        irrigation_type="drip",
        climate_zone="tropical",
        equipment=["tractor", "harvester", "irrigation_system"],
        budget_range="50k-100k",
        labor_type="mixed",
        primary_goals=["increase yield", "reduce costs"],
        current_challenges=["pest management", "water scarcity"],
        interests=["organic farming", "precision agriculture"],
        preferred_communication="text",
        language_preference="English"
    )
    return profile

def test_price_prediction_with_profile():
    """Test PricePredictionAgent with a user profile."""
    print("🧪 Testing PricePredictionAgent with user profile")
    print("=" * 50)
    
    # Create and save a test profile
    profile_manager = ProfileManager()
    test_profile = create_test_profile()
    
    print(f"📝 Creating test profile for farmer: {test_profile.name}")
    print(f"   Location: {test_profile.location}")
    print(f"   Primary crops: {test_profile.primary_crops}")
    
    # Save the profile
    success = profile_manager.save_profile(test_profile)
    if not success:
        print("❌ Failed to save test profile")
        return
    
    print("✅ Test profile saved successfully")
    
    # Test the PricePredictionAgent
    agent = PricePredictionAgent()
    
    # Test query with farmer_id
    test_query = "What's the price forecast for my crops?"
    query_with_profile = Query(
        text=test_query,
        farmer_id=test_profile.farmer_id
    )
    
    print(f"\n🔍 Testing query: '{test_query}'")
    print(f"   Farmer ID: {test_profile.farmer_id}")
    
    try:
        result = agent.process(query_with_profile)
        
        print(f"\n📊 Result:")
        print(f"   Success: {result.success}")
        print(f"   Agent: {result.agent_name}")
        print(f"   Confidence: {result.confidence}")
        
        if result.success:
            print(f"\n📈 Response:")
            print(result.response)
            
            if result.sources:
                print(f"\n📚 Sources:")
                for source in result.sources:
                    print(f"   - {source}")
        else:
            print(f"❌ Error: {result.response}")
            
    except Exception as e:
        print(f"❌ Exception occurred: {e}")
        import traceback
        traceback.print_exc()
    
    # Test query without farmer_id (should use defaults)
    print(f"\n🔍 Testing query without farmer profile:")
    query_without_profile = Query(text=test_query)
    
    try:
        result = agent.process(query_without_profile)
        
        print(f"   Success: {result.success}")
        print(f"   Agent: {result.agent_name}")
        print(f"   Confidence: {result.confidence}")
        
        if result.success:
            print(f"   Response: {result.response[:200]}...")
        else:
            print(f"   Error: {result.response}")
            
    except Exception as e:
        print(f"❌ Exception occurred: {e}")
    
    # Clean up test profile
    print(f"\n🧹 Cleaning up test profile...")
    profile_manager.delete_profile(test_profile.farmer_id)
    print("✅ Test profile deleted")

def test_profile_loading():
    """Test profile loading functionality."""
    print("\n🧪 Testing profile loading functionality")
    print("=" * 50)
    
    profile_manager = ProfileManager()
    
    # Create a test profile
    test_profile = create_test_profile()
    profile_manager.save_profile(test_profile)
    
    # Test loading the profile
    loaded_profile = profile_manager.load_profile(test_profile.farmer_id)
    
    if loaded_profile:
        print("✅ Profile loaded successfully")
        print(f"   Name: {loaded_profile.name}")
        print(f"   Location: {loaded_profile.location}")
        print(f"   Primary crops: {loaded_profile.primary_crops}")
        
        # Test location parsing
        if loaded_profile.location:
            location_parts = loaded_profile.location.split(',')
            print(f"   Location parts: {location_parts}")
            if len(location_parts) >= 2:
                district = location_parts[0].strip()
                state = location_parts[1].strip()
                print(f"   Extracted district: {district}")
                print(f"   Extracted state: {state}")
    else:
        print("❌ Failed to load profile")
    
    # Clean up
    profile_manager.delete_profile(test_profile.farmer_id)

if __name__ == "__main__":
    print("🌾 Price Prediction Agent Profile Integration Test")
    print("=" * 60)
    
    # Test profile loading first
    test_profile_loading()
    
    # Test price prediction with profile
    test_price_prediction_with_profile()
    
    print("\n✅ All tests completed!")
