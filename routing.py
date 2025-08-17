"""
Routing, orchestration, and main AI system logic for the Agricultural AI System.
"""

from typing import Dict, List, Optional, Any
from langchain_groq import ChatGroq

from utils import AgentResult, Query, GROQ_API_KEY, get_llm_response, is_farming_query
from agents import (
    BaseAgent, RishikhetAgent1, RishikhetAgent2, RishikhetAgent3, RishikhetAgent4,
    WebBasedAgent, FarmingAgent, RainForecastAgent, ImageAgent, YouTubeAgent, VisionAnalysisAgent
)

# Import LangGraph orchestrator
try:
    from langgraph_orchestrator import LangGraphAgriculturalAI
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    print("Warning: LangGraph orchestrator not available. Using traditional routing.")

# ============================================================================
# LLM ROUTER
# ============================================================================

class LLMRouter:
    def __init__(self, agents: List[BaseAgent]):
        self.agents = agents
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
        
        Examples:
        - "What's the rainfall forecast for Mumbai?" → "Mumbai"
        - "Rain prediction in New York tomorrow" → "New York" 
        - "Will it rain in Delhi next week?" → "Delhi"
        - "What's the weather like?" → "NONE"
        
        City name:"""
        
        try:
            response = self.llm.invoke(prompt)
            city = response.content.strip()
            return None if city.upper() == "NONE" else city
        except Exception as e:
            print(f"[DEBUG] AI city extraction failed: {e}")
            return None

    def select_relevant_agents(self, query_text: str) -> List[str]:
        """
        Use AI to intelligently select which agents should handle the query.
        Returns list of agent names that are relevant for the query.
        """
        # Create agent descriptions for AI to understand capabilities
        agent_descriptions = {
            "rishikhet_agent_1": "Specializes in crop diseases, pest management, and plant pathology for agricultural crops",
            "rishikhet_agent_2": "Focuses on soil management, fertilizers, nutrients, and soil health for farming",
            "rishikhet_agent_3": "Handles livestock, animal husbandry, dairy farming, and animal health issues",
            "rishikhet_agent_4": "Covers agricultural machinery, equipment, tools, and farming technology",
            "rain_forecast_agent": "Provides weather forecasts, rainfall predictions, and climate information for specific cities",
            "farming_agent": "General farming knowledge, basic agricultural advice, and farming techniques",
            "web_based_agent": "Searches current web information for farming, agriculture, and related topics",
            "image_agent": "Finds and displays images related to farming, agriculture, equipment, or any visual content",
            "youtube_agent": "Searches and retrieves YouTube videos, tutorials, demonstrations, and educational content related to farming and agriculture"
        }
        
        descriptions_text = "\n".join([f"- {name}: {desc}" for name, desc in agent_descriptions.items()])
        
        prompt = f"""
        You are an intelligent agent router for an agricultural AI system. Analyze the user query and select ONLY the most relevant agents.
        
        User Query: "{query_text}"
        
        Available Agents:
        {descriptions_text}
        
        Instructions:
        1. Select ONLY agents that are directly relevant to answering the query
        2. For image/photo requests, primarily select image_agent
        3. For weather/rainfall queries, select rain_forecast_agent
        4. For specific farming topics, select the most relevant specialized agents
        5. Don't select all agents - be selective and efficient
        6. Return agent names separated by commas
        7. If unsure, include farming_agent as a fallback
        
        Examples:
        - "show me images of tractors" → image_agent
        - "rainfall forecast for Delhi" → rain_forecast_agent
        - "tomato disease treatment" → rishikhet_agent_1, farming_agent
        - "soil pH management" → rishikhet_agent_2, farming_agent
        - "dairy cow nutrition" → rishikhet_agent_3, farming_agent
        - "best farming equipment" → rishikhet_agent_4, farming_agent, web_based_agent
        - "image of irrigation systems" → image_agent, farming_agent
        - "video tutorial on planting" → youtube_agent, farming_agent
        - "how to operate tractor" → youtube_agent, rishikhet_agent_4
        - "watch farming demonstration" → youtube_agent
        
        Selected agents (comma-separated):"""
        
        try:
            response = self.llm.invoke(prompt)
            selected_agents = [name.strip() for name in response.content.strip().split(',')]
            print(f"[DEBUG] AI selected agents: {selected_agents}")
            return selected_agents
        except Exception as e:
            print(f"[DEBUG] AI agent selection failed: {e}")
            # Fallback to basic selection
            if any(keyword in query_text.lower() for keyword in ['image', 'photo', 'picture', 'show']):
                return ['image_agent', 'farming_agent']
            elif any(keyword in query_text.lower() for keyword in ['rain', 'weather', 'forecast']):
                return ['rain_forecast_agent']
            else:
                return ['farming_agent', 'web_based_agent']

    def route_query(self, query_text: str) -> str:
        """Route the query to the best agent using AI analysis"""
        print(f"[DEBUG] LLMRouter analyzing query: '{query_text}'")
        
        # First, check if this is a weather/rainfall query with a city
        city_name = self.extract_city_from_query(query_text)
        if city_name:
            print(f"[DEBUG] City detected: {city_name}, routing to rain_forecast_agent")
            return "rain_forecast_agent"
        
        # Check for weather keywords as fallback
        weather_keywords = ["rain", "rainfall", "weather", "forecast", "precipitation", "storm", "monsoon"]
        query_lower = query_text.lower()
        if any(keyword in query_lower for keyword in weather_keywords):
            print("[DEBUG] Weather keywords detected, routing to rain_forecast_agent")
            return "rain_forecast_agent"
        
        # For other queries, use AI to determine the best agent
        agent_descriptions = []
        for agent in self.agents:
            agent_descriptions.append(f"- {agent.name}: {', '.join(agent.topics[:5])}")
        
        routing_prompt = f"""
        You are an intelligent query router for an agricultural AI system. 
        
        Query: "{query_text}"
        
        Available agents and their specialties:
        {chr(10).join(agent_descriptions)}
        
        Rules:
        1. Choose the MOST SPECIFIC agent that can handle this query
        2. If multiple agents could handle it, pick the most specialized one
        3. Return ONLY the agent name (e.g., "rishikhet_agent_2")
        4. If no specific agent fits well, return "farming_agent"
        
        Best agent name:"""
        
        try:
            response = self.llm.invoke(routing_prompt)
            selected_agent = response.content.strip()
            print(f"[DEBUG] AI router selected: {selected_agent}")
            
            # Validate the selected agent exists
            agent_names = [agent.name for agent in self.agents]
            if selected_agent in agent_names:
                return selected_agent
            else:
                print(f"[DEBUG] Invalid agent '{selected_agent}', defaulting to farming_agent")
                return "farming_agent"
                
        except Exception as e:
            print(f"[DEBUG] LLM routing failed: {e}, defaulting to farming_agent")
            return "farming_agent"

# ============================================================================
# MAIN AI SYSTEM - THE ORCHESTRATOR
# ============================================================================

class SimpleAgriculturalAI:
    def __init__(self):
        # All agents, including the new ones, are defined here.
        self.rishikhet_agents = [
            RishikhetAgent1(),
            RishikhetAgent2(),
            RishikhetAgent3(),
            RishikhetAgent4(),
            RainForecastAgent()
        ]
        self.farming_agent = FarmingAgent()
        self.web_based_agent = WebBasedAgent()
        self.image_agent = ImageAgent()
        self.youtube_agent = YouTubeAgent()
        self.vision_agent = VisionAnalysisAgent()
        self.all_agents = self.rishikhet_agents + [
            self.farming_agent,
            self.web_based_agent,
            self.image_agent,
            self.youtube_agent,
            self.vision_agent,
        ]
        
        # Initialize the new LLM-based router
        self.router = LLMRouter(self.all_agents)

    def get_llm_response(self, prompt: str) -> str:
        try:
            response = get_llm_response(prompt)
            return response
        except Exception as e:
            return f"Error: {str(e)}"

    def ask(self, question: str) -> Dict[str, Any]:
        print(f"[DEBUG] SimpleAgriculturalAI.ask() called with: '{question}'")
        query = Query(text=question)
        chain_of_thought = f"Starting single-agent routing for: '{question}'\n"

        print("[DEBUG] Using AI-driven single-agent routing...")
        # Pick exactly ONE best agent for performance
        best_agent = self.router.route_query(query.text)
        chain_of_thought += f"  - 🤖 Router chose: {best_agent}\n"
        
        return self.process_with_selected_agents(query, [best_agent], chain_of_thought)

    def analyze_image(self, image_bytes: bytes, mode: str = "plant", demo_mode: bool = False) -> Dict[str, Any]:
        """Route image analysis through the dedicated vision agent so the main system remains the entrypoint."""
        print(f"[DEBUG] SimpleAgriculturalAI.analyze_image() called, mode={mode}, demo_mode={demo_mode}")
        try:
            # Set demo mode on the vision agent
            self.vision_agent.use_demo_mode = demo_mode
            result = self.vision_agent.analyze_image(image_bytes, mode=mode)
            return {
                "answer": result.response,
                "sources": result.sources,
                "chart_path": None,
                "image_urls": [],
                "chain_of_thought": f"Processed via vision agent (mode={mode})\n",
                "agent_used": result.agent_name,
                "success": result.success,
            }
        except Exception as e:
            return {
                "answer": f"Error analyzing image: {str(e)}",
                "sources": [],
                "chart_path": None,
                "image_urls": [],
                "chain_of_thought": f"Vision agent error: {str(e)}\n",
                "agent_used": "vision_analysis_agent",
                "success": False,
            }
    
    def process_with_selected_agents(self, query: Query, selected_agent_names: List[str], chain_of_thought: str) -> Dict[str, Any]:
        """
        Process query using the AI-selected agents.
        """
        print(f"[DEBUG] Processing with selected agents: {selected_agent_names}")
        
        all_responses = []
        all_sources = []
        agents_used = []
        all_image_urls = []
        all_videos = []
        agent_details = []  # Collect per-agent outputs for UI
        
        # Create mapping of agent names to agent objects
        agent_map = {agent.name: agent for agent in self.all_agents}

        # --- Fast path: single-agent mode (no cross-agent synthesis) ---
        if len(selected_agent_names) == 1:
            agent_name = selected_agent_names[0]
            if agent_name not in agent_map:
                print(f"[DEBUG] Agent '{agent_name}' not found; falling back to farming_agent")
                agent_name = 'farming_agent'
            agent = agent_map.get(agent_name, self.farming_agent)
            chain_of_thought += f"  - 🔄 Processing only with {agent_name}\n"
            try:
                confidence, _ = agent.can_handle(query)
                print(f"[DEBUG] {agent_name} confidence: {confidence}")
                result = agent.process(query)

                # Build response directly based on agent type
                if agent_name == 'image_agent' and getattr(result, 'image_urls', None):
                    agent_details.append({
                        'name': agent_name,
                        'title': 'Images',
                        'response': '',
                        'sources': result.sources,
                        'image_urls': result.image_urls,
                        'videos': [],
                        'success': True
                    })
                    return {
                        'answer': 'Here are images related to your query.',
                        'sources': result.sources,
                        'chart_path': None,
                        'image_urls': result.image_urls,
                        'chain_of_thought': chain_of_thought,
                        'agent_used': agent_name,
                        'success': True,
                        'agent_details': agent_details
                    }
                if agent_name == 'youtube_agent' and result.data and result.data.get('videos'):
                    videos = result.data['videos']
                    agent_details.append({
                        'name': agent_name,
                        'title': 'Videos',
                        'response': '',
                        'sources': result.sources,
                        'image_urls': [],
                        'videos': videos,
                        'success': True
                    })
                    # Compose a concise textual answer listing the videos
                    answer_lines = ["Here are relevant videos:"]
                    for i, v in enumerate(videos, 1):
                        answer_lines.append(f"{i}. {v.get('title','Video')} - {v.get('url')}")
                    return {
                        'answer': "\n".join(answer_lines),
                        'sources': result.sources,
                        'chart_path': None,
                        'image_urls': [],
                        'chain_of_thought': chain_of_thought,
                        'agent_used': agent_name,
                        'success': True,
                        'agent_details': agent_details
                    }

                # Default: textual agent
                answer_text = result.response or "No answer provided."
                success = bool(result.success and (result.response or result.image_urls))
                agent_details.append({
                    'name': agent_name,
                    'title': agent_name.replace('_', ' ').title(),
                    'response': result.response or '',
                    'sources': result.sources,
                    'image_urls': getattr(result, 'image_urls', []) or [],
                    'videos': (result.data or {}).get('videos', []) if getattr(result, 'data', None) else [],
                    'success': success
                })
                return {
                    'answer': answer_text,
                    'sources': result.sources,
                    'chart_path': None,
                    'image_urls': getattr(result, 'image_urls', []) or [],
                    'chain_of_thought': chain_of_thought,
                    'agent_used': agent_name,
                    'success': success,
                    'agent_details': agent_details
                }
            except Exception as e:
                print(f"[DEBUG] {agent_name} exception (single-agent): {e}")
                agent_details.append({
                    'name': agent_name,
                    'title': 'Error',
                    'response': str(e),
                    'sources': [],
                    'image_urls': [],
                    'videos': [],
                    'success': False
                })
                return {
                    'answer': f"Error while processing with {agent_name}: {e}",
                    'sources': [],
                    'chart_path': None,
                    'image_urls': [],
                    'chain_of_thought': chain_of_thought,
                    'agent_used': agent_name,
                    'success': False,
                    'agent_details': agent_details
                }

        # Process each selected agent
        for agent_name in selected_agent_names:
            if agent_name not in agent_map:
                print(f"[DEBUG] Agent '{agent_name}' not found, skipping")
                continue
                
            agent = agent_map[agent_name]
            print(f"[DEBUG] Processing with {agent_name}...")
            chain_of_thought += f"  - 🔄 Processing with {agent_name}\n"
            
            try:
                # Check if agent can handle the query
                confidence, matching_topics = agent.can_handle(query)
                print(f"[DEBUG] {agent_name} confidence: {confidence}")
                
                if confidence > 0.1 or agent_name == 'farming_agent':  # Lower threshold, farming_agent as fallback
                    result = agent.process(query)
                    
                    if result.success and result.response and len(result.response.strip()) > 10:
                        print(f"[DEBUG] {agent_name} succeeded")
                        
                        # Handle different types of responses
                        if agent_name == 'image_agent' and result.image_urls:
                            # Store image URLs separately
                            all_image_urls.extend(result.image_urls)
                            agents_used.append(agent_name)
                            chain_of_thought += f"    - ✅ {agent_name}: Found {len(result.image_urls)} images\n"
                            agent_details.append({
                                'name': agent_name,
                                'title': 'Images',
                                'response': '',
                                'sources': result.sources,
                                'image_urls': result.image_urls,
                                'videos': [],
                                'success': True
                            })
                        elif agent_name == 'youtube_agent' and result.data and result.data.get('videos'):
                            # Handle YouTube videos separately (don't pass to LLM synthesis)
                            video_count = len(result.data['videos'])
                            agents_used.append(agent_name)
                            chain_of_thought += f"    - ✅ {agent_name}: Found {video_count} videos\n"
                            # Store videos for later appending (similar to images)
                            if not hasattr(query, 'videos'):
                                query.videos = []
                            query.videos.extend(result.data['videos'])
                            # Also collect locally for final assembly
                            all_videos.extend(result.data['videos'])
                            agent_details.append({
                                'name': agent_name,
                                'title': 'Videos',
                                'response': '',
                                'sources': result.sources,
                                'image_urls': [],
                                'videos': result.data['videos'],
                                'success': True
                            })
                        else:
                            # Regular text response
                            response_title = {
                                'farming_agent': 'General Farming Knowledge',
                                'web_based_agent': 'Current Web Information', 
                                'rain_forecast_agent': 'Weather Forecast',
                                'rishikhet_agent_1': 'Crop Disease & Pest Management',
                                'rishikhet_agent_2': 'Soil & Nutrient Management',
                                'rishikhet_agent_3': 'Livestock & Animal Husbandry',
                                'rishikhet_agent_4': 'Agricultural Equipment & Technology'
                            }.get(agent_name, f'{agent_name.title()} Knowledge')
                            
                            all_responses.append(f"**{response_title}:**\n{result.response}")
                            all_sources.extend(result.sources)
                            agents_used.append(agent_name)
                            chain_of_thought += f"    - ✅ {agent_name}: Provided specialized knowledge\n"
                            agent_details.append({
                                'name': agent_name,
                                'title': response_title,
                                'response': result.response,
                                'sources': result.sources,
                                'image_urls': [],
                                'videos': [],
                                'success': True
                            })
                    else:
                        print(f"[DEBUG] {agent_name} failed or empty response")
                        chain_of_thought += f"    - ❌ {agent_name}: No relevant information found\n"
                        agent_details.append({
                            'name': agent_name,
                            'title': 'No relevant information',
                            'response': '',
                            'sources': [],
                            'image_urls': [],
                            'videos': [],
                            'success': False
                        })
                else:
                    print(f"[DEBUG] {agent_name} skipped (low confidence: {confidence})")
                    chain_of_thought += f"    - ⚠️ {agent_name}: Query not relevant for this agent\n"
                    agent_details.append({
                        'name': agent_name,
                        'title': 'Not relevant',
                        'response': '',
                        'sources': [],
                        'image_urls': [],
                        'videos': [],
                        'success': False
                    })
                
            except Exception as e:
                print(f"[DEBUG] {agent_name} exception: {e}")
                chain_of_thought += f"    - ❌ {agent_name}: Error - {str(e)}\n"
                agent_details.append({
                    'name': agent_name,
                    'title': 'Error',
                    'response': str(e),
                    'sources': [],
                    'image_urls': [],
                    'videos': [],
                    'success': False
                })
                continue
        
        # Synthesize responses
        print(f"[DEBUG] Synthesizing {len(all_responses)} responses...")
        chain_of_thought += f"  - 🔄 Synthesizing {len(all_responses)} responses from {len(agents_used)} agents\n"
        
        if all_responses:
            # Filter out image-related responses from LLM synthesis (as implemented before)
            text_responses = [resp for resp in all_responses if not resp.startswith("**Images:**")]
            combined_response = "\n\n".join(text_responses)
            
            # Use LLM to create cohesive final answer (WITHOUT image data)
            synthesis_prompt = f"""
            You are an expert agricultural advisor. You have received multiple responses about the farming query: "{query.text}"
            
            Please synthesize these responses into a single, comprehensive, well-structured answer with the following requirements:
            
            FORMATTING REQUIREMENTS:
            - Use clear headings and subheadings with proper markdown formatting
            - Use bullet points and numbered lists for better readability
            - Add proper spacing between sections
            - Use **bold** for important terms and concepts
            - Use tables when comparing multiple items or options
            - Include emojis sparingly for visual appeal (🌾 🚜 💧 etc.)
            - Be concise: target 200–300 words total and no more than 8–12 bullet points
            
            CONTENT REQUIREMENTS:
            1. Start with a brief overview/introduction
            2. Organize information into logical sections with clear headings
            3. Combine the best information from all sources
            4. Eliminate redundancy and conflicting information
            5. Provide practical, actionable advice
            6. Include specific recommendations with dosages, timing, etc.
            7. End with key takeaways or summary points
            
            Multiple Agent Responses to Synthesize:
            {combined_response}
            
            Please provide a well-formatted, professional agricultural advisory response. Keep it concise as specified:
            """
            
            try:
                synthesized_answer = self.get_llm_response(synthesis_prompt)
                chain_of_thought += "  - 🔍 Successfully synthesized multi-agent response\n"
                
                # Append images and videos after the LLM-generated answer (if any)
                final_answer = synthesized_answer
                
                # Append images
                if all_image_urls:
                    print(f"[DEBUG] Appending {len(all_image_urls)} images to final answer")
                    final_answer += "\n\n## Related Images\n"
                    for i, image_url in enumerate(all_image_urls, 1):
                        search_term = query.text.lower().replace('image', '').replace('photo', '').replace('picture', '').strip()
                        alt_text = f"{search_term.title()} - Image {i}"
                        final_answer += f"![{alt_text}]({image_url})\n"
                
                # Append videos
                if all_videos:
                    print(f"[DEBUG] Appending {len(all_videos)} videos to final answer")
                    final_answer += "\n\n## Related Videos\n"
                    for i, video in enumerate(all_videos, 1):
                        final_answer += f"**{i}. [{video['title']}]({video['url']})**\n"
                        final_answer += f"Channel: {video['channel']}\n"
                        final_answer += f"Description: {video['description']}\n\n"
                
                return {
                    "answer": final_answer,
                    "sources": list(set(all_sources)),
                    "chart_path": None,
                    "image_urls": all_image_urls,
                    "chain_of_thought": chain_of_thought,
                    "agent_used": f"ai_selected_agents_{'+'.join(agents_used)}",
                    "success": True,
                    "agent_details": agent_details
                }
            except Exception as e:
                print(f"[DEBUG] Synthesis failed: {e}")
                # Fallback to combined response with images appended
                fallback_answer = combined_response
                if all_image_urls:
                    fallback_answer += "\n\n## Related Images\n"
                    for i, image_url in enumerate(all_image_urls, 1):
                        search_term = query.text.lower().replace('image', '').replace('photo', '').replace('picture', '').strip()
                        alt_text = f"{search_term.title()} - Image {i}"
                        fallback_answer += f"![{alt_text}]({image_url})\n"
                
                return {
                    "answer": fallback_answer,
                    "sources": list(set(all_sources)),
                    "chart_path": None,
                    "image_urls": all_image_urls,
                    "chain_of_thought": chain_of_thought,
                    "agent_used": f"ai_selected_agents_{'+'.join(agents_used)}",
                    "success": True,
                    "agent_details": agent_details
                }
        else:
            # No text responses. If we have images or videos, still return success with a brief caption.
            if all_image_urls or all_videos:
                caption = "Here are relevant visuals based on your request."
                final_answer = caption
                if all_image_urls:
                    final_answer += "\n\n## Related Images\n"
                    for i, image_url in enumerate(all_image_urls, 1):
                        search_term = query.text.lower().replace('image', '').replace('photo', '').replace('picture', '').strip()
                        alt_text = f"{search_term.title()} - Image {i}"
                        final_answer += f"![{alt_text}]({image_url})\n"
                if all_videos:
                    final_answer += "\n\n## Related Videos\n"
                    for i, video in enumerate(all_videos, 1):
                        final_answer += f"**{i}. [{video['title']}]({video['url']})**\n"
                        final_answer += f"Channel: {video['channel']}\n"
                        final_answer += f"Description: {video['description']}\n\n"
                return {
                    "answer": final_answer,
                    "sources": list(set(all_sources)),
                    "chart_path": None,
                    "image_urls": all_image_urls,
                    "chain_of_thought": chain_of_thought,
                    "agent_used": f"ai_selected_agents_{'+'.join(agents_used) if agents_used else 'visual_only'}",
                    "success": True,
                    "agent_details": agent_details
                }
            
            # Truly no content
            return {
                "answer": "I apologize, but I couldn't find relevant information for your query. Please try rephrasing your question or being more specific.",
                "sources": [],
                "chart_path": None,
                "image_urls": [],
                "chain_of_thought": chain_of_thought,
                "agent_used": "ai_selected_agents_failed",
                "success": False,
                "agent_details": agent_details
            }

        chain_of_thought += f"  - 🕵️‍♂️ AI Router selected '{best_agent.name}'.\n"
        
        print(f"[DEBUG] Starting agent processing with {best_agent.name}...")
        final_result = best_agent.process(query)
        print(f"[DEBUG] Agent processing completed. Success: {final_result.success}")
        
        chain_of_thought += f"  - ✅ {best_agent.name.title()} returned a response (Success: {final_result.success})\n"
        
        chain_of_thought += "🔍 Compiling final synthesized answer...\n"
        print("[DEBUG] Compiling final response...")

        return {
            "answer": final_result.response,
            "sources": final_result.sources,
            "chart_path": final_result.chart_path,
            "chain_of_thought": chain_of_thought,
            "agent_used": final_result.agent_name,
            "success": final_result.success
        }

    def process_farming_query_cascade(self, query: Query, chain_of_thought: str) -> Dict[str, Any]:
        """Process farming queries using cascading approach: Rishikhet → Farming → WebBased"""
        print("[DEBUG] Starting cascading agent processing...")
        chain_of_thought += "  - 🔄 Starting cascading agent approach (Rishikhet → Farming → WebBased)\n"
        
        all_responses = []
        all_sources = []
        agents_used = []
        
        # Step 1: Try Rishikhet agents first
        print("[DEBUG] Step 1: Trying Rishikhet agents...")
        chain_of_thought += "  - 📚 Step 1: Checking Rishikhet specialized agents\n"
        
        rishikhet_success = False
        for agent in self.rishikhet_agents:
            if agent.name == "rain_forecast_agent":  # Skip weather agent for farming queries
                continue
                
            try:
                confidence, matching_topics = agent.can_handle(query)
                print(f"[DEBUG] {agent.name} confidence: {confidence}")
                
                if confidence > 0.3:  # Lower threshold to try more agents
                    print(f"[DEBUG] Trying {agent.name} (confidence: {confidence})")
                    result = agent.process(query)
                    if result.success and result.response and len(result.response.strip()) > 10:
                        print(f"[DEBUG] {agent.name} succeeded with response length: {len(result.response)}")
                        all_responses.append(f"**Specialized Knowledge ({agent.name}):**\n{result.response}")
                        all_sources.extend(result.sources)
                        agents_used.append(agent.name)
                        chain_of_thought += f"    - ✅ {agent.name}: Provided specialized knowledge\n"
                        rishikhet_success = True
                        break  # Use first successful Rishikhet agent
                    else:
                        print(f"[DEBUG] {agent.name} failed or empty response")
                        chain_of_thought += f"    - ❌ {agent.name}: Failed or empty response\n"
                else:
                    print(f"[DEBUG] {agent.name} skipped (low confidence: {confidence})")
                    
            except Exception as e:
                print(f"[DEBUG] {agent.name} exception: {e}")
                chain_of_thought += f"    - ❌ {agent.name}: Error - {str(e)}\n"
                continue
        
        if not rishikhet_success:
            chain_of_thought += "    - ⚠️ No Rishikhet agents provided successful responses\n"
        
        # Step 2: ALWAYS try Farming Agent (regardless of Rishikhet success)
        print("[DEBUG] Step 2: Getting general farming knowledge...")
        chain_of_thought += "  - 🌱 Step 2: Getting general farming knowledge\n"
        
        try:
            if self.farming_agent:
                farming_result = self.farming_agent.process(query)
                if farming_result.success and farming_result.response and len(farming_result.response.strip()) > 10:
                    print(f"[DEBUG] Farming agent succeeded with response length: {len(farming_result.response)}")
                    all_responses.append(f"**General Farming Knowledge:**\n{farming_result.response}")
                    all_sources.extend(farming_result.sources)
                    agents_used.append("farming_agent")
                    chain_of_thought += "    - ✅ FarmingAgent: Provided general knowledge\n"
                else:
                    print("[DEBUG] Farming agent failed or empty response")
                    chain_of_thought += "    - ❌ FarmingAgent: Failed or empty response\n"
            else:
                print("[DEBUG] Farming agent not available")
                chain_of_thought += "    - ❌ FarmingAgent: Not available\n"
        except Exception as e:
            print(f"[DEBUG] Farming agent exception: {e}")
            chain_of_thought += f"    - ❌ FarmingAgent: Error - {str(e)}\n"
        
        # Step 3: ALWAYS try WebBased Agent (regardless of previous success)
        print("[DEBUG] Step 3: Getting current web information...")
        chain_of_thought += "  - 🌐 Step 3: Getting current web information\n"
        
        try:
            if self.web_based_agent:
                web_result = self.web_based_agent.process(query)
                print(f"[DEBUG] Web agent completed. Success: {web_result.success}")
                print(f"[DEBUG] Web agent response length: {len(web_result.response) if web_result.response else 0}")
                print(f"[DEBUG] Web agent sources: {len(web_result.sources) if web_result.sources else 0}")
                
                if web_result.success and web_result.response and len(web_result.response.strip()) > 10:
                    print("[DEBUG] Web agent succeeded with valid response")
                    all_responses.append(f"**Current Web Information:**\n{web_result.response}")
                    all_sources.extend(web_result.sources)
                    agents_used.append("web_based_agent")
                    chain_of_thought += "    - ✅ WebBasedAgent: Provided current web information\n"
                else:
                    print(f"[DEBUG] Web agent failed - Success: {web_result.success}, Response empty: {not web_result.response or len(web_result.response.strip()) <= 10}")
                    agents_used.append("web_based_agent_failed")
                    if not web_result.success:
                        chain_of_thought += "    - ❌ WebBasedAgent: Found no relevant web results\n"
                    else:
                        chain_of_thought += "    - ❌ WebBasedAgent: Returned empty/short response\n"
            else:
                print("[DEBUG] Web agent not available")
                chain_of_thought += "    - ❌ WebBasedAgent: Not available\n"
        except Exception as e:
            print(f"[DEBUG] Web agent exception: {e}")
            import traceback
            print(f"[DEBUG] Web agent traceback: {traceback.format_exc()}")
            agents_used.append("web_based_agent_error")
            chain_of_thought += f"  - ⚠️ Web agent failed with error: {str(e)}\n"
        
        # Step 4: Try ImageAgent for image-related queries
        print("[DEBUG] Step 4: Checking for image requests...")
        chain_of_thought += "  - 🖼️ Step 4: Checking for image requests\n"
        
        try:
            if self.image_agent:
                confidence, matching_topics = self.image_agent.can_handle(query)
                if confidence > 0.2:  # Lower threshold for image queries
                    print(f"[DEBUG] ImageAgent triggered with confidence: {confidence}")
                    image_result = self.image_agent.process(query)
                    if image_result.success and image_result.image_urls:
                        print(f"[DEBUG] ImageAgent succeeded with {len(image_result.image_urls)} images")
                        all_responses.append(f"**Images:**\n{image_result.response}")
                        all_sources.extend(image_result.sources)
                        agents_used.append("image_agent")
                        chain_of_thought += f"    - ✅ ImageAgent: Found {len(image_result.image_urls)} images\n"
                        
                        # Store image URLs for potential use in final result
                        if not hasattr(query, 'image_urls'):
                            query.image_urls = []
                        query.image_urls.extend(image_result.image_urls)
                    else:
                        print("[DEBUG] ImageAgent failed or no images found")
                        chain_of_thought += "    - ❌ ImageAgent: No images found\n"
                else:
                    print(f"[DEBUG] ImageAgent skipped (low confidence: {confidence})")
                    chain_of_thought += "    - ⚠️ ImageAgent: Query not image-related\n"
            else:
                print("[DEBUG] ImageAgent not available")
                chain_of_thought += "    - ❌ ImageAgent: Not available\n"
        except Exception as e:
            print(f"[DEBUG] ImageAgent exception: {e}")
            agents_used.append("image_agent_error")
            chain_of_thought += f"    - ❌ ImageAgent: Error - {str(e)}\n"
        
        # Step 5: Collect image URLs and videos from agent results
        all_image_urls = []
        all_videos = []
        
        # Collect images
        if hasattr(query, 'image_urls'):
            all_image_urls.extend(query.image_urls)
            
        # Collect videos
        if hasattr(query, 'videos'):
            all_videos.extend(query.videos)
        
        # Step 6: Synthesize all responses
        print("[DEBUG] Step 6: Synthesizing responses...")
        chain_of_thought += f"  - 🔄 Step 6: Synthesizing {len(all_responses)} responses from {len(agents_used)} agents\n"
        
        if all_responses:
            print(f"[DEBUG] Found {len(all_responses)} responses to synthesize")
            
            # Filter out image-related responses from LLM synthesis
            text_responses = []
            for response in all_responses:
                if not response.startswith("**Images:**"):
                    text_responses.append(response)
            
            print(f"[DEBUG] Filtered to {len(text_responses)} text responses (excluding images)")
            combined_response = "\n\n".join(text_responses)
            
            # Use LLM to create a cohesive final answer (WITHOUT image data)
            synthesis_prompt = f"""
            You are an expert agricultural advisor. You have received multiple responses about the farming query: "{query.text}"
            
            Please synthesize these responses into a single, comprehensive, well-structured answer with the following requirements:
            
            FORMATTING REQUIREMENTS:
            - Use clear headings and subheadings with proper markdown formatting
            - Use bullet points and numbered lists for better readability
            - Add proper spacing between sections
            - Use **bold** for important terms and concepts
            - Use tables when comparing multiple items or options
            - Include emojis sparingly for visual appeal (🌾 🚜 💧 etc.)
            
            CONTENT REQUIREMENTS:
            1. Start with a brief overview/introduction
            2. Organize information into logical sections with clear headings
            3. Combine the best information from all sources
            4. Eliminate redundancy and conflicting information
            5. Provide practical, actionable advice
            6. Include specific recommendations with dosages, timing, etc.
            7. End with key takeaways or summary points
            
            STRUCTURE EXAMPLE:
            # [Topic Title]
            
            ## Overview
            Brief introduction...
            
            ## Main Issues/Diseases
            - **Disease 1**: Description and impact
            - **Disease 2**: Description and impact
            
            ## Prevention & Treatment
            ### Chemical Control
            - Treatment option 1 with specific dosages
            - Treatment option 2 with application timing
            
            ### Cultural Practices
            - Best practice 1
            - Best practice 2
            
            ## Regional Considerations
            Location-specific advice if available...
            
            ## Key Takeaways
            - Summary point 1
            - Summary point 2
            
            Multiple Agent Responses to Synthesize:
            {combined_response}
            
            Please provide a well-formatted, professional agricultural advisory response:
            """
            
            try:
                synthesized_answer = self.get_llm_response(synthesis_prompt)
                chain_of_thought += "  - 🔍 Successfully synthesized multi-agent response\n"
                
                # Append images after the LLM-generated answer (if any)
                final_answer = synthesized_answer
                if all_image_urls:
                    print(f"[DEBUG] Appending {len(all_image_urls)} images to final answer")
                    final_answer += "\n\n## Related Images\n"
                    for i, image_url in enumerate(all_image_urls, 1):
                        # Create descriptive alt text based on search query
                        search_term = query.text.lower().replace('image', '').replace('photo', '').replace('picture', '').strip()
                        alt_text = f"{search_term.title()} - Image {i}"
                        final_answer += f"![{alt_text}]({image_url})\n"
                
                return {
                    "answer": final_answer,
                    "sources": list(set(all_sources)),  # Remove duplicates
                    "chart_path": None,
                    "image_urls": all_image_urls,  # Include image URLs in response
                    "chain_of_thought": chain_of_thought,
                    "agent_used": f"cascading_agents_{'+'.join(agents_used)}",
                    "success": True
                }
            except Exception as e:
                print(f"[DEBUG] Synthesis failed: {e}")
                # Fallback to combined response with images appended
                fallback_answer = combined_response
                if all_image_urls:
                    fallback_answer += "\n\n## Related Images\n"
                    for i, image_url in enumerate(all_image_urls, 1):
                        search_term = query.text.lower().replace('image', '').replace('photo', '').replace('picture', '').strip()
                        alt_text = f"{search_term.title()} - Image {i}"
                        fallback_answer += f"![{alt_text}]({image_url})\n"
                
                return {
                    "answer": fallback_answer,
                    "sources": list(set(all_sources)),
                    "chart_path": None,
                    "image_urls": all_image_urls,  # Include image URLs in fallback response too
                    "chain_of_thought": chain_of_thought,
                    "agent_used": f"cascading_agents_{'+'.join(agents_used)}",
                    "success": True
                }
        else:
            # No agents succeeded
            return {
                "answer": "I apologize, but I couldn't find relevant information for your farming query. Please try rephrasing your question or being more specific.",
                "sources": [],
                "chart_path": None,
                "image_urls": [],  # Empty list for failed queries
                "chain_of_thought": chain_of_thought,
                "agent_used": "cascading_agents_failed",
                "success": False
            }

# ============================================================================
# ENHANCED AI SYSTEM WITH LANGGRAPH SUPPORT
# ============================================================================

class EnhancedAgriculturalAI:
    """
    Enhanced Agricultural AI system that can use either traditional routing
    or LangGraph-based orchestration for agent collaboration.
    """
    
    def __init__(self, use_langgraph: bool = True):
        """
        Initialize the enhanced AI system.
        
        Args:
            use_langgraph: If True and available, use LangGraph orchestration.
                          If False or unavailable, use traditional routing.
        """
        self.use_langgraph = use_langgraph and LANGGRAPH_AVAILABLE
        
        if self.use_langgraph:
            print("🚀 Initializing with LangGraph orchestration...")
            self.langgraph_ai = LangGraphAgriculturalAI()
            self.traditional_ai = None
        else:
            print("📊 Initializing with traditional routing...")
            self.traditional_ai = SimpleAgriculturalAI()
            self.langgraph_ai = None
    
    def ask(self, question: str) -> Dict[str, Any]:
        """
        Ask a question using the configured AI system.
        
        Args:
            question: The user's question
            
        Returns:
            Dict containing the response and metadata
        """
        if self.use_langgraph and self.langgraph_ai:
            # Use LangGraph orchestration
            result = self.langgraph_ai.ask(question)
            
            # Convert AgentResult to the expected format
            return {
                "response": result.response,
                "agent_name": result.agent_name,
                "confidence": result.confidence,
                "sources": result.sources,
                "image_urls": result.image_urls,
                "video_urls": result.video_urls,
                "chain_of_thought": f"LangGraph orchestration used with agent: {result.agent_name}",
                "system_type": "langgraph"
            }
        else:
            # Use traditional routing
            result = self.traditional_ai.ask(question)
            result["system_type"] = "traditional"
            return result
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get information about the current system configuration."""
        return {
            "langgraph_available": LANGGRAPH_AVAILABLE,
            "using_langgraph": self.use_langgraph,
            "system_type": "langgraph" if self.use_langgraph else "traditional"
        }
    
    def switch_system(self, use_langgraph: bool = None):
        """
        Switch between LangGraph and traditional routing systems.
        
        Args:
            use_langgraph: If None, toggles current system. 
                          If True/False, switches to that system.
        """
        if use_langgraph is None:
            use_langgraph = not self.use_langgraph
        
        if use_langgraph and LANGGRAPH_AVAILABLE:
            if not self.langgraph_ai:
                print("🚀 Switching to LangGraph orchestration...")
                self.langgraph_ai = LangGraphAgriculturalAI()
            self.use_langgraph = True
            print("✅ Now using LangGraph orchestration")
        else:
            if not self.traditional_ai:
                print("📊 Switching to traditional routing...")
                self.traditional_ai = SimpleAgriculturalAI()
            self.use_langgraph = False
            print("✅ Now using traditional routing")


# ============================================================================
# DEMO AND CHAT FUNCTIONS
# ============================================================================

def demo():
    """Enhanced demo function to test both traditional and LangGraph systems."""
    print("🌾 Enhanced Agricultural AI System Demo")
    print("=" * 50)
    
    # Test both systems if LangGraph is available
    systems_to_test = []
    
    if LANGGRAPH_AVAILABLE:
        systems_to_test.append(("LangGraph", True))
        systems_to_test.append(("Traditional", False))
    else:
        systems_to_test.append(("Traditional", False))
    
    test_queries = [
        "How do I improve soil fertility for wheat crops?",
        "What's the rainfall forecast for Delhi next week?",
        "Show me images of tomato diseases",
        "Find videos about organic farming techniques"
    ]
    
    for system_name, use_langgraph in systems_to_test:
        print(f"\n🚀 Testing {system_name} System")
        print("=" * 50)
        
        ai = EnhancedAgriculturalAI(use_langgraph=use_langgraph)
        
        for query in test_queries:
            print(f"\n🔍 Query: {query}")
            print("-" * 30)
            
            try:
                result = ai.ask(query)
                print(f"System: {result.get('system_type', 'unknown')}")
                print(f"Agent: {result['agent_name']}")
                print(f"Confidence: {result['confidence']:.2f}")
                print(f"Response: {result['response'][:200]}...")
                
                if result.get('sources'):
                    print(f"Sources: {len(result['sources'])} found")
                if result.get('image_urls'):
                    print(f"Images: {len(result['image_urls'])} found")
                if result.get('video_urls'):
                    print(f"Videos: {len(result['video_urls'])} found")
                    
            except Exception as e:
                print(f"❌ Error: {str(e)}")
        
        print(f"\n✅ {system_name} system test completed")
        if len(systems_to_test) > 1:
            print("-" * 50)

def chat():
    """Enhanced interactive chat function with system switching capability."""
    print("🌱 Enhanced Agricultural AI Chat")
    print("Commands: 'quit' to exit, 'switch' to toggle systems, 'info' for system info")
    print("=" * 60)
    
    # Initialize with LangGraph if available, otherwise traditional
    ai = EnhancedAgriculturalAI(use_langgraph=True)
    
    # Show initial system info
    info = ai.get_system_info()
    print(f"🚀 Started with: {info['system_type']} system")
    print(f"LangGraph available: {info['langgraph_available']}")
    
    while True:
        try:
            user_input = input("\n🌱 You: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'bye']:
                print("👋 Goodbye! Happy farming!")
                break
            elif user_input.lower() == 'switch':
                ai.switch_system()
                continue
            elif user_input.lower() == 'info':
                info = ai.get_system_info()
                print(f"Current system: {info['system_type']}")
                print(f"LangGraph available: {info['langgraph_available']}")
                print(f"Using LangGraph: {info['using_langgraph']}")
                continue
            elif not user_input:
                continue
            
            print("🤖 AI: Processing your query...")
            result = ai.ask(user_input)
            
            print(f"\n🤖 AI ({result['agent_name']} via {result.get('system_type', 'unknown')}):")
            print(result['response'])
            
            if result.get('sources'):
                print(f"\n📚 Sources: {len(result['sources'])} found")
                for i, source in enumerate(result['sources'][:3], 1):
                    print(f"  {i}. {source}")
            
            if result.get('image_urls'):
                print(f"\n🖼️  Images: {len(result['image_urls'])} found")
            
            if result.get('video_urls'):
                print(f"\n🎥 Videos: {len(result['video_urls'])} found")
                
        except KeyboardInterrupt:
            print("\n👋 Goodbye! Happy farming!")
            break
        except Exception as e:
            print(f"❌ Error: {str(e)}")
