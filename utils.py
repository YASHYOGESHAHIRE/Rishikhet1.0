"""
Utilities, data structures, cache, and helper functions for the Agricultural AI System.
"""

import os
import json
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import google.generativeai as genai
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from urllib.robotparser import RobotFileParser
from urllib.parse import urljoin, urlparse
import time
import re

# ============================================================================
# CONFIGURATION
# Load environment variables from .env and configure API clients
# ============================================================================
load_dotenv()  # reads variables from a .env file if present

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
PIXABAY_API_KEY = os.environ.get("PIXABAY_API_KEY")
YOUTUBE_API_KEY = os.environ.get("YOUTUBE_API_KEY")
OGD_API_KEY = os.environ.get("OGD_API_KEY")

# Configure Google Generative AI only if key is present
if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)
else:
    # Leave unconfigured; downstream code should handle missing key gracefully
    pass

# Instantiate Groq LLM client (may raise if key missing; callers should ensure key is set)
llm = ChatGroq(api_key=GROQ_API_KEY, model="llama3-70b-8192", temperature=0.1)

# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class AgentResult:
    """Agent result structure with sources."""
    agent_name: str
    success: bool
    response: str
    confidence: float = 0.8
    data: Dict[str, Any] = field(default_factory=dict)
    sources: List[str] = field(default_factory=list)
    chart_path: Optional[str] = None
    image_urls: List[str] = field(default_factory=list)

@dataclass
class Query:
    """Simple query structure."""
    text: str
    intent: str = "unknown"
    confidence: float = 0.0
    farmer_id: Optional[str] = None  # Optional farmer ID for personalized responses

# ============================================================================
# CACHE SYSTEM
# ============================================================================

class SimpleCache:
    def __init__(self):
        self.cache = {}
        self.timestamps = {}
    
    def get(self, key: str, ttl_minutes: int = 60):
        if key in self.cache:
            age = (datetime.now() - self.timestamps[key]).total_seconds() / 60
            if age < ttl_minutes:
                return self.cache[key]
            else:
                del self.cache[key]
                del self.timestamps[key]
        return None
    
    def set(self, key: str, value: Any):
        self.cache[key] = value
        self.timestamps[key] = datetime.now()

# Global cache instance
cache = SimpleCache()

# ============================================================================
# TOPIC DEFINITIONS
# ============================================================================

GROUP_1_TOPICS = [
    "crop production", "seed varieties", "planting techniques", "harvesting methods",
    "yield optimization", "crop rotation", "intercropping", "monocropping"
]

GROUP_2_TOPICS = [
    "soil management", "fertilizers", "organic farming", "composting", "soil testing",
    "nutrient management", "soil health", "pH management", "soil erosion"
]

GROUP_3_TOPICS = [
    "pest control", "disease management", "integrated pest management", "pesticides",
    "biological control", "crop diseases", "insect control", "fungal diseases"
]

GROUP_4_TOPICS = [
    "irrigation", "water management", "drip irrigation", "sprinkler systems",
    "water conservation", "drainage", "water quality", "irrigation scheduling"
]

# ============================================================================
# VISUALIZATION UTILITIES
# ============================================================================

class RainfallMapVisualizer:
    """A helper class to fetch weather data and create charts."""
    
    def __init__(self):
        self.geocoding_api = "https://geocoding-api.open-meteo.com/v1/search"
        self.weather_api = "https://api.open-meteo.com/v1/forecast"

    def geocode_city(self, city_name):
        """Get coordinates for a city name with improved error handling and retries"""
        print(f"[DEBUG] Geocoding city: {city_name}")
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                params = {
                    'name': city_name,
                    'count': 5,
                    'language': 'en',
                    'format': 'json'
                }
                
                response = requests.get(self.geocoding_api, params=params, timeout=15)
                response.raise_for_status()
                data = response.json()
                
                if 'results' in data and data['results']:
                    # Try to find exact match first
                    for result in data['results']:
                        if result['name'].lower() == city_name.lower():
                            print(f"[DEBUG] Found exact match: {result['name']}")
                            return {
                                'name': result['name'],
                                'lat': result['latitude'],
                                'lon': result['longitude'],
                                'country': result.get('country', 'Unknown')
                            }
                    
                    # If no exact match, use the first result
                    result = data['results'][0]
                    print(f"[DEBUG] Using first result: {result['name']}")
                    return {
                        'name': result['name'],
                        'lat': result['latitude'],
                        'lon': result['longitude'],
                        'country': result.get('country', 'Unknown')
                    }
                else:
                    print(f"[DEBUG] No results found for {city_name}")
                    if attempt < max_retries - 1:
                        time.sleep(2)
                        continue
                    raise ValueError(f"City '{city_name}' not found")
                    
            except requests.exceptions.RequestException as e:
                print(f"[DEBUG] Geocoding attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2)
                    continue
                raise ValueError(f"Failed to geocode city '{city_name}' after {max_retries} attempts: {e}")
        
        raise ValueError(f"Failed to geocode city '{city_name}'")

    def fetch_weather_data(self, lat, lon):
        """Fetch current and forecast weather data with improved error handling"""
        print(f"[DEBUG] Fetching weather data for coordinates: {lat}, {lon}")
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                params = {
                    'latitude': lat,
                    'longitude': lon,
                    'daily': 'precipitation_sum,precipitation_probability_max',
                    'timezone': 'auto',
                    'forecast_days': 7
                }
                
                response = requests.get(self.weather_api, params=params, timeout=20)
                response.raise_for_status()
                data = response.json()
                
                print(f"[DEBUG] Weather API response keys: {list(data.keys())}")
                return data
                
            except requests.exceptions.RequestException as e:
                print(f"[DEBUG] Weather fetch attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2)
                    continue
                raise ValueError(f"Failed to fetch weather data after {max_retries} attempts: {e}")
        
        raise ValueError("Failed to fetch weather data")

    def get_rainfall_intensity(self, rainfall):
        """Categorize rainfall intensity"""
        if rainfall == 0:
            return "No rain"
        elif rainfall < 2.5:
            return "Light rain"
        elif rainfall < 10:
            return "Moderate rain"
        elif rainfall < 35:
            return "Heavy rain"
        else:
            return "Very heavy rain"

    def create_daily_forecast_chart(self, weather_data, city_name):
        """Create a daily rainfall forecast chart"""
        print("[DEBUG] Creating daily forecast chart...")
        
        try:
            daily_data = weather_data.get('daily', {})
            dates = daily_data.get('time', [])
            precipitation = daily_data.get('precipitation_sum', [])
            
            if not dates or not precipitation:
                print("[DEBUG] No daily data available for chart")
                return None
            
            # Create DataFrame
            df = pd.DataFrame({
                'Date': pd.to_datetime(dates),
                'Precipitation (mm)': precipitation
            })
            
            # Create the chart
            fig = px.bar(
                df, 
                x='Date', 
                y='Precipitation (mm)',
                title=f'7-Day Rainfall Forecast for {city_name}',
                color='Precipitation (mm)',
                color_continuous_scale='Blues'
            )
            
            fig.update_layout(
                xaxis_title="Date",
                yaxis_title="Precipitation (mm)",
                showlegend=False,
                height=400
            )
            
            print("[DEBUG] Chart created successfully")
            return fig
            
        except Exception as e:
            print(f"[DEBUG] Chart creation failed: {e}")
            return None

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_llm_response(prompt: str) -> str:
    """Get response from LLM with caching"""
    try:
        response = llm.invoke(prompt)
        return response.content
    except Exception as e:
        return f"Error: {str(e)}"

def is_farming_query(question: str) -> bool:
    """Check if the query is farming-related"""
    farming_keywords = [
        "farming", "agriculture", "crop", "livestock", "irrigation", "fertilizer",
        "pesticide", "harvest", "planting", "soil", "cultivation", "organic",
        "sustainable", "yield", "seeds", "disease", "pest", "nutrition",
        "breeding", "dairy", "poultry", "aquaculture", "mechanization"
    ]
    question_lower = question.lower()
    return any(keyword in question_lower for keyword in farming_keywords)
