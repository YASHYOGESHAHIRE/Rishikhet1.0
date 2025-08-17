"""
Agent classes and implementations for the Agricultural AI System.
"""

import os
import requests
import time
import base64
from io import BytesIO
from typing import Dict, List, Optional, Any
from langchain_groq import ChatGroq
from urllib.robotparser import RobotFileParser
from urllib.parse import urljoin, urlparse
import re
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
import holidays

from utils import (
    AgentResult, Query, cache, GROQ_API_KEY, PIXABAY_API_KEY, YOUTUBE_API_KEY, OGD_API_KEY,
    GROUP_1_TOPICS, GROUP_2_TOPICS, GROUP_3_TOPICS, GROUP_4_TOPICS,
    RainfallMapVisualizer, get_llm_response
)

# ============================================================================
# BASE AGENT CLASS
# ============================================================================

class BaseAgent:
    """Base class for all agents. Just inherit and implement process()."""
    
    def __init__(self, name: str, topics: List[str], description: Optional[str] = None):
        self.name = name
        self.topics = topics
        self.description = description or ""

    def can_handle(self, query: Query):
        """Return confidence (0-1) and matching topics if this agent can handle the query.

        Strategy:
        - First try lightweight keyword/topic overlap (fast path)
        - If no overlap and a description exists, ask LLM to rate relevance 0-1
        """
        query_lower = query.text.lower()
        matching_topics = []
        
        for topic in self.topics:
            if topic.lower() in query_lower:
                matching_topics.append(topic)
        
        if matching_topics:
            confidence = min(len(matching_topics) * 0.3, 1.0)
            return confidence, matching_topics

        # Fallback to LLM-based relevance if we have a description
        if self.description:
            try:
                prompt = (
                    "You are a strict classifier.\n"
                    f"Agent Name: {self.name}\n"
                    f"Agent Description: {self.description}\n\n"
                    f"User Query: {query.text}\n\n"
                    "On a scale from 0.0 to 1.0, how relevant is this agent to answer the query?\n"
                    "Respond with ONLY a decimal number between 0 and 1 (e.g., 0.0, 0.25, 0.7, 1.0)."
                )
                raw = self.get_llm_response(prompt)
                # Extract first float-like number
                m = re.search(r"\d*\.\d+|\d+", raw)
                if m:
                    score = float(m.group(0))
                    # normalize/clamp
                    if score > 1:
                        score = score / 100 if score > 10 else 1.0
                    score = max(0.0, min(1.0, score))
                else:
                    score = 0.0
                return score, matching_topics
            except Exception:
                # Safe fallback if LLM relevance fails
                return 0.0, matching_topics

        # Default: no match
        return 0.0, matching_topics

    def process(self, query: Query) -> AgentResult:
        """
        Process the query and return a structured AgentResult.
        Override this method in each specific agent.
        """
        return AgentResult(
            agent_name=self.name,
            success=False,
            response="This agent has not implemented the process method.",
            confidence=0.0
        )

    def get_llm_response(self, prompt: str) -> str:
        """Helper to get cached LLM response."""
        cache_key = f"llm_response_{hash(prompt)}"
        cached = cache.get(cache_key, ttl_minutes=30)
        if cached:
            return cached
        
        try:
            response = get_llm_response(prompt)
            cache.set(cache_key, response)
            return response
        except Exception as e:
            return f"Error getting LLM response: {str(e)}"

    def get_llm_response_with_sources(self, prompt: str) -> tuple:
        """
        New helper method to get a structured LLM response
        including both the answer and a list of sources.
        """
        response = self.get_llm_response(prompt)
        
        # For now, we'll extract sources from the response if they exist
        # This is a simple implementation - you might want to enhance it
        sources = []
        lines = response.split('\n')
        for line in lines:
            if line.strip().startswith('Source:') or line.strip().startswith('Reference:'):
                sources.append(line.strip().replace('Source:', '').replace('Reference:', '').strip())
        
        return response, sources

# ============================================================================
# RISHIKHET AGENTS
# ============================================================================

class RishikhetAgent1(BaseAgent):
    def __init__(self):
        super().__init__(
            "rishikhet_agent_1",
            topics=GROUP_1_TOPICS,
            description=(
                "Expert in crop production, seed varieties, planting techniques, harvesting methods, "
                "and yield optimization strategies. Focuses on practical, actionable crop husbandry guidance."
            ),
        )

    def process(self, query: Query) -> AgentResult:
        prompt = f"""
        You are an expert in crop production, seed varieties, planting techniques, and harvesting methods.
        Please provide detailed, practical advice for the following farming query: {query.text}
        
        Focus areas:
        1) Crop production techniques
        2) Seed selection and varieties
        3) Planting methods and timing
        4) Harvesting best practices
        5) Yield optimization strategies
        
        Provide actionable advice that farmers can implement.
        """
        
        response = self.get_llm_response(prompt)
        return AgentResult(
            agent_name=self.name,
            success=True,
            response=response,
            confidence=0.8,
            sources=["Rishikhet Agricultural Knowledge Base - Crop Production"]
        )

class RishikhetAgent2(BaseAgent):
    def __init__(self):
        super().__init__(
            "rishikhet_agent_2",
            topics=GROUP_2_TOPICS,
            description=(
                "Expert in soil management, fertilizers (organic and synthetic), composting, soil testing, "
                "nutrient and pH management, soil health improvement, and erosion control."
            ),
        )

    def process(self, query: Query) -> AgentResult:
        prompt = f"""
        You are an expert in soil management, fertilizers, and organic farming practices.
        Please provide detailed, practical advice for the following farming query: {query.text}
        
        Focus areas:
        1) Soil health and management
        2) Fertilizer recommendations and application
        3) Organic farming techniques
        4) Composting methods
        5) Nutrient management strategies
        
        Provide specific recommendations with dosages and timing where applicable.
        """
        
        response = self.get_llm_response(prompt)
        return AgentResult(
            agent_name=self.name,
            success=True,
            response=response,
            confidence=0.8,
            sources=["Rishikhet Agricultural Knowledge Base - Soil & Fertilizers"]
        )

class RishikhetAgent3(BaseAgent):
    def __init__(self):
        super().__init__(
            "rishikhet_agent_3",
            topics=GROUP_3_TOPICS,
            description=(
                "Expert in pest and disease management, Integrated Pest Management (IPM), biological control, "
                "pesticide selection and safe use, and diagnosis of crop diseases (insect, fungal, bacterial)."
            ),
        )

    def process(self, query: Query) -> AgentResult:
        prompt = f"""
        You are an expert in pest control, disease management, and integrated pest management.
        Please provide detailed, practical advice for the following farming query: {query.text}
        
        Focus areas:
        1) Pest identification and control methods
        2) Disease prevention and treatment
        3) Integrated pest management strategies
        4) Biological control methods
        5) Safe pesticide use and alternatives
        
        Provide specific treatment recommendations with application rates and safety precautions.
        """
        
        response = self.get_llm_response(prompt)
        return AgentResult(
            agent_name=self.name,
            success=True,
            response=response,
            confidence=0.8,
            sources=["Rishikhet Agricultural Knowledge Base - Pest & Disease Management"]
        )

class RishikhetAgent4(BaseAgent):
    def __init__(self):
        super().__init__(
            "rishikhet_agent_4",
            topics=GROUP_4_TOPICS,
            description=(
                "Expert in irrigation and water management including drip/sprinkler systems, scheduling, "
                "conservation, drainage, and water quality considerations."
            ),
        )

    def process(self, query: Query) -> AgentResult:
        prompt = f"""
        You are an expert in irrigation, water management, and water conservation techniques.
        Please provide detailed, practical advice for the following farming query: {query.text}
        
        Focus areas:
        1) Irrigation system design and management
        2) Water conservation techniques
        3) Drainage solutions
        4) Water quality management
        5) Efficient irrigation scheduling
        
        Provide practical solutions for water management challenges.
        """
        
        response = self.get_llm_response(prompt)
        return AgentResult(
            agent_name=self.name,
            success=True,
            response=response,
            confidence=0.8,
            sources=["Rishikhet Agricultural Knowledge Base - Water Management"]
        )

# ============================================================================
# SPECIALIZED AGENTS
# ============================================================================

class WebBasedAgent(BaseAgent):
    """Web-based agent that searches for farming information using DuckDuckGo
    and checks robots.txt compliance for each result."""
    
    def __init__(self):
        super().__init__("web_based_agent", topics=[
            "farming", "agriculture", "crop", "livestock", "irrigation", "fertilizer",
            "pesticide", "harvest", "planting", "soil", "weather", "market", "prices",
            "techniques", "modern farming", "organic farming", "sustainable agriculture"
        ])
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })

    def check_robots_txt(self, url: str) -> bool:
        """Check if the URL is allowed by robots.txt"""
        try:
            parsed_url = urlparse(url)
            robots_url = f"{parsed_url.scheme}://{parsed_url.netloc}/robots.txt"
            
            rp = RobotFileParser()
            rp.set_url(robots_url)
            rp.read()
            
            return rp.can_fetch(self.session.headers.get('User-Agent', '*'), url)
        except Exception as e:
            print(f"[DEBUG] Robots.txt check failed for {url}: {e}")
            return True  # If we can't check, assume it's allowed

    def search_duckduckgo(self, query: str, max_results: int = 5) -> List[Dict]:
        """Search DuckDuckGo for farming-related information"""
        try:
            # Add farming context to the query
            enhanced_query = f"{query} farming agriculture"
            
            search_url = "https://api.duckduckgo.com/"
            params = {
                'q': enhanced_query,
                'format': 'json',
                'no_redirect': '1',
                'no_html': '1',
                'skip_disambig': '1'
            }
            
            print(f"[DEBUG] Searching DuckDuckGo for: {query}")
            response = self.session.get(search_url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            print(f"[DEBUG] DuckDuckGo API response keys: {list(data.keys())}")
            print(f"[DEBUG] Abstract available: {bool(data.get('Abstract'))}")
            print(f"[DEBUG] RelatedTopics count: {len(data.get('RelatedTopics', []))}")
            
            results = []
            
            # Get instant answer if available
            if data.get('Abstract'):
                abstract_result = {
                    'title': data.get('Heading', 'DuckDuckGo Instant Answer'),
                    'url': data.get('AbstractURL', 'https://duckduckgo.com'),
                    'snippet': data.get('Abstract', ''),
                    'source': data.get('AbstractSource', 'DuckDuckGo')
                }
                results.append(abstract_result)
                print(f"[DEBUG] Added abstract result: {abstract_result['title'][:50]}...")
            
            # Get related topics
            for i, topic in enumerate(data.get('RelatedTopics', [])[:max_results-1]):
                if isinstance(topic, dict) and topic.get('Text'):
                    topic_result = {
                        'title': topic.get('Text', '')[:100] + '...',
                        'url': topic.get('FirstURL', ''),
                        'snippet': topic.get('Text', ''),
                        'source': 'DuckDuckGo Related Topic'
                    }
                    results.append(topic_result)
                    print(f"[DEBUG] Added related topic {i+1}: {topic_result['title'][:50]}...")

            # Also parse direct Results list if present
            ddg_results = data.get('Results', []) or []
            for i, item in enumerate(ddg_results[:max_results]):
                if isinstance(item, dict) and (item.get('FirstURL') or item.get('URL')):
                    url = item.get('FirstURL') or item.get('URL')
                    title = item.get('Text') or item.get('Title') or url
                    snippet = item.get('Text') or ''
                    results.append({
                        'title': title[:120],
                        'url': url,
                        'snippet': snippet,
                        'source': 'DuckDuckGo Result'
                    })
                    print(f"[DEBUG] Added result {i+1}: {title[:50]}...")
            
            print(f"[DEBUG] Total results before robots.txt filtering: {len(results)}")
            
            # Filter results based on robots.txt
            allowed_results = []
            for result in results:
                if result['url'] and self.check_robots_txt(result['url']):
                    allowed_results.append(result)
                    print(f"[DEBUG] Allowed by robots.txt: {result['url']}")
                else:
                    print(f"[DEBUG] Blocked by robots.txt: {result['url']}")
            
            print(f"[DEBUG] Final allowed results: {len(allowed_results)}")
            return allowed_results[:max_results]
            
        except Exception as e:
            print(f"[DEBUG] DuckDuckGo search failed: {e}")
            return []
    
    def fetch_page_content(self, url: str) -> str:
        """Fetch and extract text content from a webpage"""
        try:
            response = self.session.get(url, timeout=15)
            response.raise_for_status()
            
            # Simple text extraction (remove HTML tags)
            content = response.text
            content = re.sub(r'<[^>]+>', '', content)
            content = re.sub(r'\s+', ' ', content).strip()
            
            # Limit content length
            return content[:2000] if len(content) > 2000 else content
            
        except Exception as e:
            print(f"[DEBUG] Failed to fetch content from {url}: {e}")
            return ""

    def process(self, query: Query) -> AgentResult:
        """Process farming query using web search"""
        print(f"[DEBUG] WebBasedAgent processing: {query.text}")
        
        # Search for relevant information
        search_results = self.search_duckduckgo(query.text)
        print(f"[DEBUG] WebBasedAgent got {len(search_results) if search_results else 0} search results")
        
        if not search_results:
            print("[DEBUG] WebBasedAgent: No search results found, returning failure")
            return AgentResult(
                agent_name=self.name,
                success=False,
                response="No relevant web results found for your farming query.",
                confidence=0.3,
                sources=[]
            )
        
        # Compile information from search results
        compiled_info = []
        sources = []
        
        for result in search_results:
            if result['snippet']:
                compiled_info.append(f"**{result['title']}**\n{result['snippet']}")
                sources.append(result['url'])

        # If we still don't have any compiled info, try fetching page content of a few results
        if not compiled_info:
            print("[DEBUG] No snippets available, fetching page content for context...")
            for result in search_results[:3]:
                try:
                    content = self.fetch_page_content(result['url'])
                    if content:
                        compiled_info.append(f"**{result['title']}**\n{content[:500]}...")
                        sources.append(result['url'])
                        print(f"[DEBUG] Fetched content from: {result['url']}")
                except Exception as e:
                    print(f"[DEBUG] Failed to fetch content for {result['url']}: {e}")

        # Use LLM to synthesize the information
        context = "\n\n".join(compiled_info)
        synthesis_prompt = f"""
        Based on the following web search results about farming/agriculture, provide a comprehensive answer to the user's question: "{query.text}"
        
        Web Search Results:
        {context}
        
        Please provide a detailed, practical answer that would be helpful for farmers. Focus on actionable advice and current best practices.
        """
        
        try:
            synthesized_response = self.get_llm_response(synthesis_prompt)
            
            return AgentResult(
                agent_name=self.name,
                success=True,
                response=synthesized_response,
                confidence=0.8,
                sources=sources
            )
            
        except Exception as e:
            print(f"[DEBUG] LLM synthesis failed: {e}")
            # Fallback to raw search results
            fallback_response = "Based on web search results:\n\n" + "\n\n".join(compiled_info)
            
            return AgentResult(
                agent_name=self.name,
                success=True,
                response=fallback_response,
                confidence=0.6,
                sources=sources
            )

class FarmingAgent(BaseAgent):
    """Enhanced farming agent that always collaborates with Rishikhet agents
    for comprehensive agricultural expertise."""
    
    def __init__(self):
        super().__init__("farming_agent", topics=["general farming", "how to", "what is", "explain", "help", "guide", "tips", "farming", "crop", "disease"])
        # Initialize Rishikhet agents for collaboration
        self.rishikhet_agents = [
            RishikhetAgent1(),
            RishikhetAgent2(), 
            RishikhetAgent3(),
            RishikhetAgent4()
        ]
        # Initialize ImageAgent for visual content
        self.image_agent = ImageAgent()

    def process(self, query: Query) -> AgentResult:
        """Process query using FarmingAgent, Rishikhet agents, and ImageAgent for comprehensive coverage."""
        
        # First, get responses from all Rishikhet agents
        rishikhet_responses = []
        all_sources = ["General Farming Knowledge"]
        image_urls = []
        
        for agent in self.rishikhet_agents:
            try:
                # Check if this Rishikhet agent can handle the query
                confidence, matching_topics = agent.can_handle(query)
                if confidence > 0.2:  # Use agent if it has some relevance
                    result = agent.process(query)
                    if result.success:
                        rishikhet_responses.append({
                            'agent': agent.name,
                            'response': result.response,
                            'topics': matching_topics,
                            'confidence': result.confidence
                        })
                        # Add sources from Rishikhet agents
                        if result.sources:
                            all_sources.extend(result.sources)
            except Exception as e:
                print(f"Error with {agent.name}: {str(e)}")
                continue
        
        # Check if ImageAgent can provide relevant images
        try:
            image_confidence, image_topics = self.image_agent.can_handle(query)
            if image_confidence > 0.1:  # Lower threshold for images as they're supplementary
                image_result = self.image_agent.process(query)
                if image_result.success and hasattr(image_result, 'image_urls') and image_result.image_urls:
                    image_urls = image_result.image_urls
                    if image_result.sources:
                        all_sources.extend(image_result.sources)
                    print(f"ImageAgent found {len(image_urls)} images for the query")
        except Exception as e:
            print(f"Error with ImageAgent: {str(e)}")
        
        # Generate comprehensive response using Rishikhet insights
        if rishikhet_responses:
            # Create a synthesis prompt that combines Rishikhet expertise
            rishikhet_insights = "\n\n".join([
                f"**{resp['agent']} (Topics: {', '.join(resp['topics'])}):**\n{resp['response']}"
                for resp in rishikhet_responses
            ])
            
            synthesis_prompt = f"""
            You are a comprehensive farming assistant. You have received specialized insights from Rishikhet agricultural experts.
            
            Original Question: {query.text}
            
            Specialized Expert Insights:
            {rishikhet_insights}
            
            Based on these expert insights and your general farming knowledge, provide a comprehensive, well-structured answer that:
            1. Synthesizes the best information from all sources
            2. Provides practical, actionable advice
            3. Addresses the specific question asked
            4. Maintains accuracy and relevance
            
            If the expert insights don't fully address the question, supplement with your general farming knowledge.
            """
            
            response = self.get_llm_response(synthesis_prompt)
            confidence = min(0.9, max([resp['confidence'] for resp in rishikhet_responses]) + 0.1)
            
        else:
            # Fallback to general farming knowledge if no Rishikhet agents can help
            prompt = f"You are a knowledgeable farming assistant. Please provide helpful, practical advice for this farming question: {query.text}"
            response = self.get_llm_response(prompt)
            confidence = 0.7
        
        return AgentResult(
            agent_name=f"{self.name} (with Rishikhet collaboration)",
            success=True, 
            response=response, 
            confidence=confidence,
            sources=list(set(all_sources)),  # Remove duplicates
            image_urls=image_urls  # Include images for direct display
        )

class RainForecastAgent(BaseAgent):
    def __init__(self):
        topics = ["rain", "rainfall", "forecast", "precipitation", "weather"]
        super().__init__("rain_forecast_agent", topics=topics)
        self.visualizer = RainfallMapVisualizer()
        self.llm = ChatGroq(api_key=GROQ_API_KEY, model="llama3-70b-8192", temperature=0.1, timeout=10)

    def extract_city_from_query(self, query_text: str) -> Optional[str]:
        """
        Use AI to extract city name from rainfall/weather related queries.
        Returns the city name if found, None otherwise.
        """
        prompt = f"""
        Extract the city name from the following query. The query is related to rainfall, weather, or forecast.
        
        Query: "{query_text}"
        
        Rules:
        1. Return ONLY the city name if you can identify one
        2. If multiple cities are mentioned, return the primary/main city
        3. If no city is mentioned or you're unsure, return "NONE"
        4. Return just the city name without any additional text or explanation
        5. Handle variations like "New York", "NYC", "New Delhi", etc.
        6. Do not include the characters '=' or '-' in your output
        
        Examples:
        Example 1: "What's the rainfall forecast for Mumbai?" → "Mumbai"
        Example 2: "Rain prediction in New York tomorrow" → "New York"
        Example 3: "Will it rain in Delhi next week?" → "Delhi"
        Example 4: "What's the weather like?" → "NONE"
        
        City name:"""
        
        try:
            response = self.llm.invoke(prompt)
            city = response.content.strip()
            return None if city.upper() == "NONE" else city
        except Exception as e:
            print(f"[DEBUG] AI city extraction failed: {e}")
            return None

    def process(self, query: Query) -> AgentResult:
        print(f"[DEBUG] RainForecastAgent processing: {query.text}")
        
        # Extract city using AI
        city_name = self.extract_city_from_query(query.text)
        print(f"[DEBUG] AI extracted city: {city_name}")
        
        if not city_name:
            return AgentResult(
                agent_name=self.name,
                success=False,
                response="I couldn't identify a city in your rainfall query. Please specify a city name for the forecast.",
                confidence=0.3,
                sources=[]
            )
        
        try:
            # Get city coordinates
            city_data = self.visualizer.geocode_city(city_name)
            print(f"[DEBUG] Geocoded city data: {city_data}")
            
            # Fetch weather data
            weather_data = self.visualizer.fetch_weather_data(city_data['lat'], city_data['lon'])
            print(f"[DEBUG] Weather data keys: {list(weather_data.keys())}")
            
            # Create chart (temporarily disabled to avoid hangs)
            chart_path = None
            # daily_chart = self.visualizer.create_daily_forecast_chart(weather_data, city_data['name'])
            # if daily_chart:
            #     chart_filename = f"rainfall_forecast_{city_data['name'].replace(' ', '_').lower()}.png"
            #     chart_path = os.path.join("static", "charts", chart_filename)
            #     os.makedirs(os.path.dirname(chart_path), exist_ok=True)
            #     pio.write_image(daily_chart, chart_path)
            
            # Generate text response
            daily_data = weather_data.get('daily', {})
            forecast_times = daily_data.get('time', [])
            precipitation_sums = daily_data.get('precipitation_sum', [])
            
            forecast_lines = [
                f"🌧️ **7-Day Rainfall Forecast for {city_data['name']}, {city_data['country']}**\n"
            ]
            
            if not forecast_times or not precipitation_sums:
                forecast_lines.append("⚠️ No forecast data available for this location.")
            else:
                for i in range(min(7, len(forecast_times))):
                    date = forecast_times[i]
                    rain_amount = precipitation_sums[i]
                    intensity = self.visualizer.get_rainfall_intensity(rain_amount)
                    forecast_lines.append(f"• **{date}**: {rain_amount:.1f} mm ({intensity}).")
            
            response = "\n".join(forecast_lines)
            
            return AgentResult(
                agent_name=self.name,
                success=True,
                response=response,
                confidence=0.9,
                chart_path=chart_path,
                sources=[f"Open-Meteo Weather API for {city_data['name']}"]
            )
            
        except Exception as e:
            print(f"[DEBUG] RainForecastAgent error: {e}")
            return AgentResult(
                agent_name=self.name,
                success=False,
                response=f"Sorry, I couldn't fetch the rainfall forecast for {city_name}. Error: {str(e)}",
                confidence=0.2,
                sources=[]
            )


# ============================================================================
# IMAGE AGENT
# ============================================================================

class ImageAgent(BaseAgent):
    """Agent for searching and retrieving images using Pixabay API."""
    
    def __init__(self):
        topics = ["image", "photo", "picture", "show me", "find image", "search for image", "display image"]
        super().__init__("image_agent", topics=topics)
        self.api_key = PIXABAY_API_KEY
        self.api_url = "https://pixabay.com/api/"
    
    def download_and_encode_image(self, image_url: str, max_size_kb: int = 500) -> Optional[str]:
        """
        Download an image and convert it to base64 data URL for direct embedding.
        Returns None if download fails or image is too large.
        """
        try:
            print(f"[DEBUG] Downloading image: {image_url}")
            response = requests.get(image_url, timeout=10, stream=True)
            response.raise_for_status()
            
            # Check content length if available
            content_length = response.headers.get('content-length')
            if content_length and int(content_length) > max_size_kb * 1024:
                print(f"[DEBUG] Image too large: {content_length} bytes")
                return None
            
            # Read image data
            image_data = BytesIO()
            downloaded_size = 0
            for chunk in response.iter_content(chunk_size=8192):
                downloaded_size += len(chunk)
                if downloaded_size > max_size_kb * 1024:  # Stop if too large
                    print(f"[DEBUG] Image too large during download: {downloaded_size} bytes")
                    return None
                image_data.write(chunk)
            
            # Get content type
            content_type = response.headers.get('content-type', 'image/jpeg')
            if not content_type.startswith('image/'):
                content_type = 'image/jpeg'  # Default fallback
            
            # Convert to base64
            image_data.seek(0)
            base64_data = base64.b64encode(image_data.getvalue()).decode('utf-8')
            data_url = f"data:{content_type};base64,{base64_data}"
            
            print(f"[DEBUG] Successfully encoded image: {len(base64_data)} characters")
            return data_url
            
        except Exception as e:
            print(f"[DEBUG] Failed to download/encode image {image_url}: {e}")
            return None
        
    def process(self, query: Query) -> AgentResult:
        """Process image search query and return results with image URLs."""
        keywords_to_remove = self.topics + ["of", "a", "an"]
        search_term = query.text.lower()
        for keyword in keywords_to_remove:
            search_term = search_term.replace(keyword, "")
        search_term = search_term.strip()

        if not self.api_key:
            return AgentResult(
                agent_name=self.name,
                success=False,
                response="Pixabay API key not configured.",
                confidence=1.0
            )
            
        if not search_term:
            return AgentResult(
                agent_name=self.name,
                success=False,
                response="Please provide a search term for the image.",
                confidence=1.0
            )

        params = {
            'key': self.api_key,
            'q': search_term,
            'image_type': 'photo',
            'per_page': 3  # Get a few results
        }
        
        try:
            response = requests.get(self.api_url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if data['hits']:
                image_urls = [hit['webformatURL'] for hit in data['hits']]
                image_tags = [f"[{hit['tags']}]" for hit in data['hits']]
                
                # Download and convert images to base64 for direct display
                base64_images = []
                successful_downloads = 0
                
                for i, url in enumerate(image_urls):
                    print(f"[DEBUG] Processing image {i+1}/{len(image_urls)}")
                    base64_data = self.download_and_encode_image(url)
                    if base64_data:
                        base64_images.append(base64_data)
                        successful_downloads += 1
                    else:
                        # Fallback to original URL if base64 conversion fails
                        base64_images.append(url)
                
                response_text = f"Found {len(image_urls)} images for '{search_term}' ({successful_downloads} downloaded for direct display):"
                
                return AgentResult(
                    agent_name=self.name,
                    success=True,
                    response=response_text,
                    confidence=0.9,
                    image_urls=base64_images,  # Now contains base64 data URLs or fallback URLs
                    sources=[f"Pixabay API - Search: '{search_term}'"]
                )
            else:
                return AgentResult(
                    agent_name=self.name,
                    success=False,
                    response=f"No images found for '{search_term}'.",
                    confidence=0.9
                )
                
        except requests.exceptions.RequestException as e:
            return AgentResult(
                agent_name=self.name,
                success=False,
                response=f"Error fetching images: {str(e)}",
                confidence=0.0
            )


# ============================================================================
# VISION ANALYSIS AGENT (Plant/Soil via Gemini Vision)
# ============================================================================

class VisionAnalysisAgent(BaseAgent):
    """Analyze plant disease/health and soil characteristics from an image using Gemini Vision.

    Requires environment variable GEMINI_API_KEY to be set (in `utils.py` you can store and load it).
    If not set, the agent will return an informative error.
    """

    def __init__(self):
        topics = [
            "plant", "leaf", "disease", "blight", "rust", "spots", "deficiency",
            "soil", "nutrient", "texture", "pH", "fertility", "organic matter",
            "plant analysis", "soil analysis", "vision", "image"
        ]
        super().__init__("vision_analysis_agent", topics=topics, description=(
            "Analyzes uploaded plant or soil images to detect plant diseases, deficiencies, and provide soil observations using a vision LLM."
        ))
        self.gemini_api_key = os.environ.get("GEMINI_API_KEY", "")
        self.gemini_model = "gemini-1.5-flash"  # fast and cost-effective
        self.gemini_url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.gemini_model}:generateContent"

    def _build_prompt(self, mode: str) -> str:
        if mode == "soil":
            return (
                "You are an expert agronomist. Analyze the soil image provided and describe: "
                "1) Visible texture and structure (sand/silt/clay indications), "
                "2) Moisture and organic matter indications, 3) Possible nutrient or pH hints, "
                "4) Practical recommendations: soil tests to run, amendments/fertilizers, irrigation, and field actions. "
                "Be specific, concise, and actionable. If uncertain, state assumptions."
            )
        # default to plant
        return (
            "You are a plant pathologist. Analyze the plant/leaf image provided and describe: "
            "1) Likely disease/deficiency/pest (with confidence), 2) Identifying symptoms, "
            "3) Immediate actions and safe treatments (organic + chemical options with dosage), "
            "4) Prevention and care tips. Be specific and practical for farmers."
        )

    def analyze_image(self, image_bytes: bytes, mode: str = "plant") -> AgentResult:
        if not self.gemini_api_key:
            return AgentResult(
                agent_name=self.name,
                success=False,
                response=(
                    "Gemini API key not configured. Set GEMINI_API_KEY in environment to enable vision analysis."
                ),
                confidence=0.0,
                sources=[]
            )

        try:
            b64 = base64.b64encode(image_bytes).decode("utf-8")
            prompt = self._build_prompt(mode)

            payload = {
                "contents": [
                    {
                        "parts": [
                            {"text": prompt},
                            {
                                "inline_data": {
                                    "mime_type": "image/jpeg",
                                    "data": b64
                                }
                            }
                        ]
                    }
                ],
                "generationConfig": {
                    "temperature": 0.2,
                    "maxOutputTokens": 512
                }
            }

            params = {"key": self.gemini_api_key}
            resp = requests.post(self.gemini_url, params=params, json=payload, timeout=30)
            resp.raise_for_status()
            data = resp.json()

            # Extract text
            answer = ""
            try:
                candidates = data.get("candidates", [])
                if candidates and "content" in candidates[0]:
                    parts = candidates[0]["content"].get("parts", [])
                    answer = "".join(p.get("text", "") for p in parts)
            except Exception:
                answer = str(data)[:1000]

            if not answer:
                answer = "No analysis text returned by the vision model."

            return AgentResult(
                agent_name=self.name,
                success=True,
                response=answer,
                confidence=0.85,
                sources=[f"{self.gemini_model} (Gemini Vision)"]
            )
        except Exception as e:
            return AgentResult(
                agent_name=self.name,
                success=False,
                response=f"Vision analysis failed: {str(e)}",
                confidence=0.0,
                sources=[]
            )

    def process(self, query: Query) -> AgentResult:
        # Not used for text-only route; image endpoint will call analyze_image directly.
        return AgentResult(
            agent_name=self.name,
            success=False,
            response="Provide an image via the /analyze-image endpoint for analysis.",
            confidence=0.0,
            sources=[]
        )


# ============================================================================
# YOUTUBE AGENT
# ============================================================================

class YouTubeAgent(BaseAgent):
    """Agent for searching and retrieving YouTube videos using YouTube Data API."""
    
    def __init__(self):
        topics = ["video", "youtube", "tutorial", "how to", "demonstration", "watch", "learn", "guide"]
        super().__init__("youtube_agent", topics=topics)
        self.api_key = YOUTUBE_API_KEY
        self.api_url = "https://www.googleapis.com/youtube/v3/search"
        
    def process(self, query: Query) -> AgentResult:
        """Process video search query and return results with video URLs."""
        keywords_to_remove = self.topics + ["video", "youtube", "of", "a", "an", "the"]
        search_term = query.text.lower()
        for keyword in keywords_to_remove:
            search_term = search_term.replace(keyword, "")
        search_term = search_term.strip()
        
        # Add farming context to search if not already present
        farming_keywords = ["farming", "agriculture", "crop", "livestock", "irrigation", "tractor"]
        if not any(keyword in search_term.lower() for keyword in farming_keywords):
            search_term += " farming agriculture"

        if not self.api_key:
            return AgentResult(
                agent_name=self.name,
                success=False,
                response="YouTube API key not configured.",
                confidence=1.0
            )
            
        if not search_term:
            return AgentResult(
                agent_name=self.name,
                success=False,
                response="Please provide a search term for the video.",
                confidence=1.0
            )

        params = {
            'key': self.api_key,
            'q': search_term,
            'part': 'snippet',
            'type': 'video',
            'maxResults': 5,  # Get top 5 videos
            'order': 'relevance',
            'videoDefinition': 'any',
            'videoDuration': 'any'
        }
        
        try:
            print(f"[DEBUG] Searching YouTube for: '{search_term}'")
            response = requests.get(self.api_url, params=params, timeout=10)
            print(f"[DEBUG] YouTube API response status: {response.status_code}")
            
            if response.status_code == 403:
                print(f"[DEBUG] YouTube API quota exceeded or permissions issue")
                return AgentResult(
                    agent_name=self.name,
                    success=False,
                    response="YouTube API quota exceeded or API key lacks permissions. Please check YouTube Data API setup.",
                    confidence=0.0
                )
            
            response.raise_for_status()
            data = response.json()
            print(f"[DEBUG] YouTube API response keys: {list(data.keys())}")
            
            if 'error' in data:
                print(f"[DEBUG] YouTube API error: {data['error']}")
                return AgentResult(
                    agent_name=self.name,
                    success=False,
                    response=f"YouTube API error: {data['error'].get('message', 'Unknown error')}",
                    confidence=0.0
                )
            
            if data.get('items'):
                videos = []
                video_urls = []
                
                print(f"[DEBUG] Found {len(data['items'])} YouTube videos")
                for item in data['items']:
                    video_id = item['id']['videoId']
                    video_url = f"https://www.youtube.com/watch?v={video_id}"
                    title = item['snippet']['title']
                    description = item['snippet']['description'][:200] + "..." if len(item['snippet']['description']) > 200 else item['snippet']['description']
                    channel = item['snippet']['channelTitle']
                    
                    videos.append({
                        'title': title,
                        'description': description,
                        'channel': channel,
                        'url': video_url
                    })
                    video_urls.append(video_url)
                
                # Create response text with video details
                response_text = f"Found {len(videos)} YouTube videos for '{search_term}':\n\n"
                for i, video in enumerate(videos, 1):
                    response_text += f"**{i}. {video['title']}**\n"
                    response_text += f"Channel: {video['channel']}\n"
                    response_text += f"Description: {video['description']}\n"
                    response_text += f"URL: {video['url']}\n\n"
                
                print(f"[DEBUG] YouTube agent returning {len(videos)} videos")
                return AgentResult(
                    agent_name=self.name,
                    success=True,
                    response=response_text,
                    confidence=0.9,
                    data={'videos': videos},
                    sources=[f"YouTube API - Search: '{search_term}'"]
                )
            else:
                print(f"[DEBUG] No items in YouTube API response")
                print(f"[DEBUG] Full response: {data}")
                return AgentResult(
                    agent_name=self.name,
                    success=False,
                    response=f"No YouTube videos found for '{search_term}'. API response: {data}",
                    confidence=0.9
                )
                
        except requests.exceptions.RequestException as e:
            return AgentResult(
                agent_name=self.name,
                success=False,
                response=f"Error fetching YouTube videos: {str(e)}",
                confidence=0.0
            )
        except KeyError as e:
            return AgentResult(
                agent_name=self.name,
                success=False,
                response=f"Error parsing YouTube API response: {str(e)}",
                confidence=0.0
            )


# ============================================================================
# PRICE PREDICTION AGENT
# ============================================================================

class PricePredictionAgent(BaseAgent):
    def __init__(self):
        topics = ["price", "prediction", "forecast", "market", "commodity", "prices", "trend", "analysis"]
        super().__init__("price_prediction_agent", topics=topics, description=(
            "Predicts agricultural commodity prices using historical data and Prophet forecasting model."
        ))
        self.API_KEY = OGD_API_KEY
        self.API1_URL = "https://api.data.gov.in/resource/35985678-0d79-46b4-9ed6-6f13308a1d24"
        self.API2_URL = "https://api.data.gov.in/resource/9ef84268-d588-465a-a308-a864a43d0070"
        self.VISUALIZER_PATH = "/tmp/price_prediction_charts"
        
        # Commodity mapping for common variations
        self.commodity_mapping = {
            "corn": ["Maize", "Corn", "Sweet Corn"],
            "maize": ["Maize", "Corn", "Sweet Corn"],
            "wheat": ["Wheat", "Atta"],
            "rice": ["Rice", "Paddy", "Basmati"],
            "onion": ["Onion", "Pyaz"],
            "tomato": ["Tomato", "Tamatar"],
            "potato": ["Potato", "Aloo"],
            "sugar": ["Sugar", "Gur", "Jaggery"],
            "cotton": ["Cotton", "Kapas"],
            "soybean": ["Soybean", "Soya"],
            "pulses": ["Pulses", "Dal", "Lentils"],
            "milk": ["Milk", "Doodh"],
            "eggs": ["Eggs", "Anda"]
        }

    def find_best_commodity_match(self, requested_commodity: str) -> str:
        """Find the best commodity match from the mapping or return the original."""
        requested_lower = requested_commodity.lower()
        
        # Check if we have a mapping for this commodity
        if requested_lower in self.commodity_mapping:
            return self.commodity_mapping[requested_lower][0]  # Return the first (most common) variant
        
        # If no mapping found, return the original (capitalized)
        return requested_commodity.title()

    def fetch_data(self, api_url, filters, limit=50000):
        """Fetch raw data from API with a given set of filters."""
        params = {
            "api-key": self.API_KEY,
            "format": "json",
            "limit": limit
        }
        params.update(filters)

        try:
            response = requests.get(api_url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()

            if 'records' in data and data['records']:
                print(f"✅ {api_url} fetched {len(data['records'])} records.")
                return pd.DataFrame(data['records'])
            else:
                print(f"⚠️ {api_url} returned no data.")
                return None
        except requests.exceptions.Timeout:
            print(f"❌ {api_url} request timed out.")
        except requests.exceptions.RequestException as e:
            print(f"❌ {api_url} error: {e}")
        return None

    def get_holidays(self, years=[2023, 2024, 2025]):
        """Generate a DataFrame of Indian holidays and Sundays."""
        holiday_df = pd.DataFrame([])
        for date, name in sorted(holidays.India(years=years).items()):
            holiday_df = pd.concat([
                holiday_df,
                pd.DataFrame({'ds': [date], 'holiday': [f"India-Holidays: {name}"]})
            ], ignore_index=True)

        start_date = pd.to_datetime(f'{years[0]}-01-01')
        end_date = pd.to_datetime(f'{years[-1]}-12-31')
        sundays = pd.date_range(start=start_date, end=end_date, freq='W-SUN')
        sunday_holidays = pd.DataFrame({'ds': sundays, 'holiday': 'Sunday'})
        
        holiday_df = pd.concat([holiday_df, sunday_holidays], ignore_index=True)
        holiday_df['ds'] = pd.to_datetime(holiday_df['ds'])
        return holiday_df

    def process_data_and_forecast(self, state, district, commodity, market, forecast_days=7):
        """
        Main function to process data, train the model, and generate a price forecast.
        """
        # --- 1. Fetch Data ---
        # Try to find the best commodity match
        mapped_commodity = self.find_best_commodity_match(commodity)
        print(f"🔍 Looking for commodity: '{commodity}' -> mapped to: '{mapped_commodity}'")
        
        api_params1 = {"filters[State]": state, "filters[District]": district, "filters[Commodity]": mapped_commodity}
        df = self.fetch_data(self.API1_URL, api_params1)
        
        if df is None or df.empty:
            print("ℹ️ Trying API 2...")
            api_params2 = {"filters[state.keyword]": state, "filters[district]": district, "filters[commodity]": mapped_commodity}
            df = self.fetch_data(self.API2_URL, api_params2, limit=50000)

        if df is None or df.empty:
            # Try alternative commodity names from mapping
            if commodity.lower() in self.commodity_mapping:
                alternatives = self.commodity_mapping[commodity.lower()][1:]  # Skip the first one we already tried
                for alt_commodity in alternatives:
                    print(f"🔄 Trying alternative commodity name: '{alt_commodity}'")
                    api_params1 = {"filters[State]": state, "filters[District]": district, "filters[Commodity]": alt_commodity}
                    df = self.fetch_data(self.API1_URL, api_params1)
                    if df is not None and not df.empty:
                        mapped_commodity = alt_commodity
                        break
                    
                    if df is None or df.empty:
                        api_params2 = {"filters[state.keyword]": state, "filters[district]": district, "filters[commodity]": alt_commodity}
                        df = self.fetch_data(self.API2_URL, api_params2, limit=50000)
                        if df is not None and not df.empty:
                            mapped_commodity = alt_commodity
                            break

        if df is None or df.empty:
            return f"❌ Failed to fetch data for '{commodity}' (tried: {mapped_commodity} and alternatives).", None, None

        # --- 2. Filter Data ---
        df.columns = [col.lower().replace('.', '_') for col in df.columns]
        
        required_cols = ["market", "variety", "grade", "modal_price", "arrival_date"]
        if not all(col in df.columns for col in required_cols):
            return "⚠️ Required columns not found in data after normalization.", None, None

        df_filtered = df[
            (df["market"].str.lower().eq(market.lower())) 
            # (df["variety"].str.lower().eq(variety.lower())) &
            # (df["grade"].str.lower().eq(grade.lower()))
        ].copy()  # Create a copy to avoid SettingWithCopyWarning

        if df_filtered.empty:
            return "⚠️ No data after filtering.", None, None
        
        # --- 3. Prepare for Prophet model ---
        df_filtered.loc[:, "arrival_date"] = pd.to_datetime(df_filtered["arrival_date"], errors='coerce')
        df_filtered = df_filtered.dropna(subset=["arrival_date", "modal_price"])
        df_filtered.loc[:, "modal_price"] = pd.to_numeric(df_filtered["modal_price"], errors='coerce')
        df_filtered = df_filtered.dropna(subset=["modal_price"])
        df_filtered = df_filtered.sort_values("arrival_date")

        # Handle duplicate dates by taking the mean price for each date
        df_filtered = df_filtered.groupby("arrival_date")["modal_price"].mean().reset_index()
        
        # Ensure modal_price is numeric after grouping
        df_filtered["modal_price"] = pd.to_numeric(df_filtered["modal_price"], errors='coerce')
        df_filtered = df_filtered.dropna(subset=["modal_price"])
        
        prophet_df = df_filtered[["arrival_date", "modal_price"]].rename(columns={"arrival_date": "ds", "modal_price": "y"})
        prophet_df = prophet_df.set_index("ds").asfreq("D").interpolate(method="linear").reset_index()

        # --- 4. Get holidays ---
        holiday_df = self.get_holidays()

        # --- 5. Prophet Model and Forecast ---
        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
            seasonality_mode="additive",
            changepoint_prior_scale=0.5,
            holidays=holiday_df
        )
        model.fit(prophet_df)
        
        # Create future dataframe starting from today for the next 7 days
        from datetime import datetime, timedelta
        today = datetime.now().date()
        future_dates = [today + timedelta(days=i) for i in range(forecast_days)]
        future = pd.DataFrame({'ds': future_dates})
        
        forecast = model.predict(future)

        # --- 6. Generate plots ---
        os.makedirs(self.VISUALIZER_PATH, exist_ok=True)
        
        plot_path_forecast = os.path.join(self.VISUALIZER_PATH, "price_forecast.png")
        fig1 = model.plot(forecast)
        plt.title("Price Forecast", fontsize=14)
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.grid(True)
        plt.savefig(plot_path_forecast)
        plt.close(fig1)

        plot_path_components = os.path.join(self.VISUALIZER_PATH, "price_components.png")
        fig2 = model.plot_components(forecast)
        plt.savefig(plot_path_components)
        plt.close(fig2)

        return forecast, [plot_path_forecast, plot_path_components], mapped_commodity

    def process(self, query: Query) -> AgentResult:
        """Process price prediction query and return forecast results."""
        
        # Initialize parameters with defaults
        params = {
            "state": "Maharashtra",
            "district": "Dhule", 
            "commodity": "Onion",
            "market": "Dhule"
        }
        
        # Try to get parameters from user profile first
        if query.farmer_id:
            try:
                from profile_models import ProfileManager
                profile_manager = ProfileManager()
                profile = profile_manager.load_profile(query.farmer_id)
                
                if profile and profile.location:
                    # Extract state and district from location
                    location_parts = profile.location.split(',')
                    if len(location_parts) >= 2:
                        params["district"] = location_parts[0].strip()
                        params["state"] = location_parts[1].strip()
                        params["market"] = location_parts[0].strip()  # Use district as market
                
                # Use primary crops from profile if available
                if profile.primary_crops:
                    params["commodity"] = profile.primary_crops[0]  # Use first primary crop
                    
            except Exception as e:
                print(f"Error loading farmer profile: {e}")
        
        # Extract parameters from query using LLM
        extraction_prompt = f"""
        Extract the following parameters from this price prediction query: {query.text}
        
        Return ONLY a JSON object with these fields:
        - state: The state name (default: Maharashtra)
        - district: The district name (default: Dhule)
        - commodity: The commodity name (REQUIRED - extract from query)
        - market: The market name (default: Dhule)
        
        IMPORTANT: You MUST extract the commodity name from the query. Common commodities: corn, maize, wheat, rice, onion, tomato, potato, sugar, cotton, soybean, pulses, milk, eggs.
        
        If any parameter is not mentioned, use "NONE" as the value.
        Example: {{"state": "Maharashtra", "district": "Dhule", "commodity": "corn", "market": "Dhule"}}
        """
        
        try:
            response = self.get_llm_response(extraction_prompt)
            # Simple JSON extraction - you might want to enhance this
            import json
            import re
            
            # Find JSON-like structure in response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                try:
                    extracted_params = json.loads(json_match.group())
                    
                    # Update params with extracted values, keeping profile defaults for "NONE" values
                    for key, value in extracted_params.items():
                        if value != "NONE":
                            params[key] = value
                except json.JSONDecodeError:
                    print(f"Failed to parse JSON from LLM response: {json_match.group()}")
                    # Continue with default params if JSON parsing fails
            
            # Process the forecast - only pass the parameters the method expects
            forecast_params = {
                'state': params.get('state'),
                'district': params.get('district'),
                'commodity': params.get('commodity'),
                'market': params.get('market'),
                'forecast_days': 7
            }
            
            # Validate that commodity is not empty
            if not forecast_params['commodity'] or forecast_params['commodity'] == 'NONE':
                return AgentResult(
                    agent_name=self.name,
                    success=False,
                    response="❌ Could not identify the commodity from your query. Please specify a commodity like 'corn', 'wheat', 'onion', etc.",
                    confidence=0.0,
                    sources=[]
                )
            
            try:
                forecast, chart_paths, mapped_commodity = self.process_data_and_forecast(**forecast_params)
            except ValueError as e:
                if "too many values to unpack" in str(e):
                    # Handle the case where the method returns more than 3 values
                    result = self.process_data_and_forecast(**forecast_params)
                    if isinstance(result, tuple) and len(result) >= 3:
                        forecast, chart_paths, mapped_commodity = result[0], result[1], result[2]
                    else:
                        return AgentResult(
                            agent_name=self.name,
                            success=False,
                            response=f"❌ Error processing forecast: {str(e)}",
                            confidence=0.0,
                            sources=[]
                        )
                else:
                    raise
            
            if isinstance(forecast, str):  # Error message
                return AgentResult(
                    agent_name=self.name,
                    success=False,
                    response=forecast,
                    confidence=0.0,
                    sources=[]
                )
            
            # Format the response
            last_7_days = forecast[['ds', 'yhat']].tail(7)
            # Use the mapped commodity name in the response
            display_commodity = mapped_commodity if mapped_commodity else params['commodity']
            forecast_text = f"📊 **Price Forecast for {display_commodity} in {params['market']}, {params['district']}, {params['state']}**\n\n"
            forecast_text += "**Next 7 Days Forecast:**\n"
            
            for _, row in last_7_days.iterrows():
                date = row['ds'].strftime('%Y-%m-%d')
                price = row['yhat']
                forecast_text += f"• **{date}**: ₹{price:.2f}\n"
            
            # Use the first chart path as the main chart_path
            main_chart_path = chart_paths[0] if chart_paths else None
            # forecast_text += f"\n📈 Charts saved at: {', '.join(chart_paths) if chart_paths else 'No charts generated'}"
            
            return AgentResult(
                agent_name=self.name,
                success=True,
                response=forecast_text,
                confidence=0.8,
                chart_path=main_chart_path,
                sources=[f"Data.gov.in API - {params['commodity']} prices"]
            )
            
        except Exception as e:
            return AgentResult(
                agent_name=self.name,
                success=False,
                response=f"Error processing price prediction: {str(e)}",
                confidence=0.0,
                sources=[]
            )
