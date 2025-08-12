from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
import json
import os

class FarmerProfile(BaseModel):
    """Comprehensive farmer profile model for personalized agricultural assistance"""
    
    # Basic Information
    farmer_id: str = Field(..., description="Unique identifier for the farmer")
    name: str = Field(..., description="Farmer's full name")
    email: Optional[str] = Field(None, description="Contact email")
    phone: Optional[str] = Field(None, description="Contact phone number")
    
    # Location Information
    location: Optional[str] = Field(None, description="Farm location (city, state, country)")
    latitude: Optional[float] = Field(None, description="Farm latitude for weather data")
    longitude: Optional[float] = Field(None, description="Farm longitude for weather data")
    farm_size_acres: Optional[float] = Field(None, description="Total farm size in acres")
    
    # Farming Details
    primary_crops: List[str] = Field(default_factory=list, description="Main crops grown")
    secondary_crops: List[str] = Field(default_factory=list, description="Secondary/rotation crops")
    livestock: List[str] = Field(default_factory=list, description="Types of livestock")
    farming_type: Optional[str] = Field(None, description="Organic, conventional, mixed, etc.")
    experience_years: Optional[int] = Field(None, description="Years of farming experience")
    
    # Soil and Environment
    soil_type: Optional[str] = Field(None, description="Primary soil type")
    irrigation_type: Optional[str] = Field(None, description="Irrigation method used")
    climate_zone: Optional[str] = Field(None, description="Climate zone/region")
    
    # Equipment and Resources
    equipment: List[str] = Field(default_factory=list, description="Farm equipment owned")
    budget_range: Optional[str] = Field(None, description="Annual farming budget range")
    labor_type: Optional[str] = Field(None, description="Family, hired, mixed labor")
    
    # Goals and Challenges
    primary_goals: List[str] = Field(default_factory=list, description="Main farming objectives")
    current_challenges: List[str] = Field(default_factory=list, description="Current farming challenges")
    interests: List[str] = Field(default_factory=list, description="Areas of interest for learning")
    
    # Preferences
    preferred_communication: Optional[str] = Field(None, description="Preferred communication style")
    language_preference: Optional[str] = Field("English", description="Preferred language")
    
    # Metadata
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    profile_completeness: float = Field(default=0.0, description="Profile completion percentage")

    def calculate_completeness(self) -> float:
        """Calculate profile completeness percentage"""
        total_fields = 0
        filled_fields = 0
        
        # Core fields (weighted more heavily)
        core_fields = ['name', 'location', 'primary_crops', 'farming_type', 'farm_size_acres']
        for field in core_fields:
            total_fields += 2  # Weight core fields as 2
            value = getattr(self, field)
            if value and (not isinstance(value, list) or len(value) > 0):
                filled_fields += 2
        
        # Optional fields
        optional_fields = ['email', 'phone', 'secondary_crops', 'livestock', 'soil_type', 
                          'irrigation_type', 'equipment', 'primary_goals', 'current_challenges']
        for field in optional_fields:
            total_fields += 1
            value = getattr(self, field)
            if value and (not isinstance(value, list) or len(value) > 0):
                filled_fields += 1
        
        return round((filled_fields / total_fields) * 100, 1) if total_fields > 0 else 0.0

    def get_context_for_ai(self) -> str:
        """Generate context string for AI personalization"""
        context_parts = []
        
        if self.name:
            context_parts.append(f"Farmer: {self.name}")
        
        if self.location:
            context_parts.append(f"Location: {self.location}")
        
        if self.primary_crops:
            context_parts.append(f"Primary crops: {', '.join(self.primary_crops)}")
        
        if self.farm_size_acres:
            context_parts.append(f"Farm size: {self.farm_size_acres} acres")
        
        if self.farming_type:
            context_parts.append(f"Farming type: {self.farming_type}")
        
        if self.experience_years:
            context_parts.append(f"Experience: {self.experience_years} years")
        
        if self.soil_type:
            context_parts.append(f"Soil type: {self.soil_type}")
        
        if self.current_challenges:
            context_parts.append(f"Current challenges: {', '.join(self.current_challenges)}")
        
        if self.primary_goals:
            context_parts.append(f"Goals: {', '.join(self.primary_goals)}")
        
        return "; ".join(context_parts)

class ProfileUpdateRequest(BaseModel):
    """Request model for updating farmer profile"""
    farmer_id: str
    updates: Dict[str, Any]

class ProfileManager:
    """Manager class for farmer profile operations"""
    
    def __init__(self, data_dir: str = "profile_data"):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
    
    def _get_profile_path(self, farmer_id: str) -> str:
        """Get file path for farmer profile"""
        return os.path.join(self.data_dir, f"{farmer_id}.json")
    
    def save_profile(self, profile: FarmerProfile) -> bool:
        """Save farmer profile to file"""
        try:
            profile.updated_at = datetime.now()
            profile.profile_completeness = profile.calculate_completeness()
            
            profile_path = self._get_profile_path(profile.farmer_id)
            with open(profile_path, 'w', encoding='utf-8') as f:
                json.dump(profile.dict(), f, indent=2, default=str)
            return True
        except Exception as e:
            print(f"Error saving profile: {e}")
            return False
    
    def load_profile(self, farmer_id: str) -> Optional[FarmerProfile]:
        """Load farmer profile from file"""
        try:
            profile_path = self._get_profile_path(farmer_id)
            if not os.path.exists(profile_path):
                return None
            
            with open(profile_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Convert string dates back to datetime objects
            if 'created_at' in data and isinstance(data['created_at'], str):
                data['created_at'] = datetime.fromisoformat(data['created_at'].replace('Z', '+00:00'))
            if 'updated_at' in data and isinstance(data['updated_at'], str):
                data['updated_at'] = datetime.fromisoformat(data['updated_at'].replace('Z', '+00:00'))
            
            return FarmerProfile(**data)
        except Exception as e:
            print(f"Error loading profile: {e}")
            return None
    
    def update_profile(self, farmer_id: str, updates: Dict[str, Any]) -> Optional[FarmerProfile]:
        """Update existing farmer profile"""
        profile = self.load_profile(farmer_id)
        if not profile:
            return None
        
        # Update fields
        for key, value in updates.items():
            if hasattr(profile, key):
                setattr(profile, key, value)
        
        if self.save_profile(profile):
            return profile
        return None
    
    def delete_profile(self, farmer_id: str) -> bool:
        """Delete farmer profile"""
        try:
            profile_path = self._get_profile_path(farmer_id)
            if os.path.exists(profile_path):
                os.remove(profile_path)
            return True
        except Exception as e:
            print(f"Error deleting profile: {e}")
            return False
    
    def list_profiles(self) -> List[str]:
        """List all farmer profile IDs"""
        try:
            profiles = []
            for filename in os.listdir(self.data_dir):
                if filename.endswith('.json'):
                    profiles.append(filename[:-5])  # Remove .json extension
            return profiles
        except Exception as e:
            print(f"Error listing profiles: {e}")
            return []
