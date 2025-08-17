#!/usr/bin/env python3
"""
Test script to verify profile update functionality works correctly.
"""

import os
import sys
import json
from datetime import datetime

# Add the current directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from profile_models import FarmerProfile, ProfileManager

def test_profile_update():
    """Test profile update functionality with various data types."""
    print("🧪 Testing Profile Update Functionality")
    print("=" * 50)
    
    # Create profile manager
    profile_manager = ProfileManager()
    
    # Create a test profile
    test_profile = FarmerProfile(
        farmer_id="test_update_001",
        name="Test Farmer",
        email="test@example.com",
        phone="1234567890",
        location="Dhule, Maharashtra, India",
        latitude=20.9024,
        longitude=74.7774,
        farm_size_acres=50.0,
        primary_crops=["Onion", "Tomato"],
        secondary_crops=["Corn", "Soybeans"],
        livestock=[],
        farming_type="conventional",
        experience_years=15,
        soil_type="loamy",
        irrigation_type="drip",
        climate_zone="tropical",
        equipment=["tractor", "harvester"],
        budget_range="50k-100k",
        labor_type="mixed",
        primary_goals=["increase yield", "reduce costs"],
        current_challenges=["pest management", "water scarcity"],
        interests=["organic farming", "precision agriculture"],
        preferred_communication="text",
        language_preference="English"
    )
    
    # Save the initial profile
    print("📝 Creating initial test profile...")
    success = profile_manager.save_profile(test_profile)
    if not success:
        print("❌ Failed to save initial profile")
        return
    
    print("✅ Initial profile saved successfully")
    
    # Test 1: Update with string values that should be converted to numbers
    print("\n🔍 Test 1: Updating numeric fields with string values")
    updates_1 = {
        "experience_years": "10",  # String that should become int
        "farm_size_acres": "75.5",  # String that should become float
        "latitude": "21.1234",  # String that should become float
        "longitude": "75.5678"  # String that should become float
    }
    
    updated_profile = profile_manager.update_profile(test_profile.farmer_id, updates_1)
    if updated_profile:
        print("✅ Profile updated successfully")
        print(f"   Experience years: {updated_profile.experience_years} (type: {type(updated_profile.experience_years)})")
        print(f"   Farm size: {updated_profile.farm_size_acres} (type: {type(updated_profile.farm_size_acres)})")
        print(f"   Latitude: {updated_profile.latitude} (type: {type(updated_profile.latitude)})")
        print(f"   Longitude: {updated_profile.longitude} (type: {type(updated_profile.longitude)})")
    else:
        print("❌ Failed to update profile")
    
    # Test 2: Update with string values that should be converted to lists
    print("\n🔍 Test 2: Updating list fields with comma-separated strings")
    updates_2 = {
        "primary_crops": "Wheat, Rice, Cotton",  # String that should become list
        "equipment": "tractor, harvester, irrigation_system, drone",  # String that should become list
        "primary_goals": "increase yield, reduce costs, improve soil health"  # String that should become list
    }
    
    updated_profile = profile_manager.update_profile(test_profile.farmer_id, updates_2)
    if updated_profile:
        print("✅ Profile updated successfully")
        print(f"   Primary crops: {updated_profile.primary_crops} (type: {type(updated_profile.primary_crops)})")
        print(f"   Equipment: {updated_profile.equipment} (type: {type(updated_profile.equipment)})")
        print(f"   Primary goals: {updated_profile.primary_goals} (type: {type(updated_profile.primary_goals)})")
    else:
        print("❌ Failed to update profile")
    
    # Test 3: Update with invalid numeric values (should be skipped)
    print("\n🔍 Test 3: Updating with invalid numeric values (should be skipped)")
    updates_3 = {
        "experience_years": "invalid_number",  # Invalid string that should be skipped
        "farm_size_acres": "not_a_number",  # Invalid string that should be skipped
        "name": "Updated Farmer Name"  # Valid string that should work
    }
    
    updated_profile = profile_manager.update_profile(test_profile.farmer_id, updates_3)
    if updated_profile:
        print("✅ Profile updated successfully")
        print(f"   Name: {updated_profile.name}")
        print(f"   Experience years: {updated_profile.experience_years} (should remain unchanged)")
        print(f"   Farm size: {updated_profile.farm_size_acres} (should remain unchanged)")
    else:
        print("❌ Failed to update profile")
    
    # Test 4: Test the get_context_for_ai method
    print("\n🔍 Test 4: Testing AI context generation")
    context = updated_profile.get_context_for_ai()
    print(f"   AI Context: {context}")
    
    # Clean up
    print(f"\n🧹 Cleaning up test profile...")
    profile_manager.delete_profile(test_profile.farmer_id)
    print("✅ Test profile deleted")

if __name__ == "__main__":
    test_profile_update()
    print("\n✅ All profile update tests completed!")

