"""
FIXED: Routing, orchestration, and main AI system logic for the Agricultural AI System.
"""

import os
from typing import Dict, List, Optional, Any
from langchain_groq import ChatGroq

# Import utilities and agents
from utils import AgentResult, Query, GROQ_API_KEY, get_llm_response, is_farming_query
from agents import (
    BaseAgent, RishikhetAgent1, RishikhetAgent2, RishikhetAgent3, RishikhetAgent4,
    WebBasedAgent, FarmingAgent, RainForecastAgent
)

# ============================================================================
# FIXED LLM ROUTER
# ============================================================================

class LLMRouter:
    def __init__(self, agents: List[BaseAgent]):
        self.agents = agents
        # Fixed: Add error handling for LLM initialization
        try:
            self.llm = ChatGroq(
                api_key=GROQ_API_KEY, 
                model="llama3-70b-8192", 
                temperature=0.1, 
                timeout=10
            )
            print("✅ LLM Router initialized successfully")
        except Exception as e:
            print(f"❌ LLM Router initialization failed: {e}")
            self.llm = None

    def extract_city_from_query(self, query_text: str) -> Optional[str]:
        """
        Use AI to extract city name from rainfall/weather related queries.
        Returns the city name if found, None otherwise.
        """
        if not self.llm:
            print("⚠️ LLM not available, skipping city extraction")
            return None
            
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
            
            if city.upper() == "NONE" or not city:
                return None
            
            print(f"🏙️ AI extracted city: '{city}'")
            return city
            
        except Exception as e:
            print(f"❌ City extraction failed: {e}")
            return None

    def route_query(self, query_text: str) -> str:
        """
        Route the query to the best agent using AI analysis.
        Returns the agent name that should handle the query.
        """
        print(f"🔀 Routing query: '{query_text[:50]}...'")
        
        # First, check for city extraction for weather queries
        if any(keyword in query_text.lower() for keyword in ["rain", "weather", "forecast", "precipitation"]):
            city_name = self.extract_city_from_query(query_text)
            if city_name:
                print(f"🌧️ Weather query with city detected, routing to rain_forecast_agent")
                return "rain_forecast_agent"
        
        # Fallback keyword-based routing for weather queries
        weather_keywords = ["rain", "weather", "forecast", "precipitation", "temperature", "climate"]
        if any(keyword in query_text.lower() for keyword in weather_keywords):
            print("🌦️ Weather query detected (fallback), routing to rain_forecast_agent")
            return "rain_forecast_agent"
        
        # For other queries, use AI routing if available
        if self.llm:
            try:
                agent_descriptions = {
                    "rishikhet_agent_1": "Crop management, planting, harvesting, general agriculture",
                    "rishikhet_agent_2": "Soil management, fertilizers, soil testing, nutrients",
                    "rishikhet_agent_3": "Pest control, disease management, plant protection",
                    "rishikhet_agent_4": "Organic farming, sustainable practices, soil testing",
                    "farming_agent": "General farming advice, agricultural practices",
                    "web_based_agent": "Latest farming information from web sources"
                }
                
                descriptions_text = "\n".join([f"- {name}: {desc}" for name, desc in agent_descriptions.items()])
                
                prompt = f"""
                Given the following query about farming/agriculture, select the BEST agent to handle it.
                
                Query: "{query_text}"
                
                Available agents:
                {descriptions_text}
                
                Rules:
                1. Return ONLY the agent name (e.g., "rishikhet_agent_2")
                2. Choose the most specific agent that matches the query topic
                3. If unsure, default to "farming_agent"
                
                Best agent:"""
                
                response = self.llm.invoke(prompt)
                agent_name = response.content.strip()
                
                if agent_name in agent_descriptions:
                    print(f"🤖 AI selected agent: '{agent_name}'")
                    return agent_name
                else:
                    print(f"⚠️ AI returned invalid agent '{agent_name}', defaulting to farming_agent")
                    return "farming_agent"
                    
            except Exception as e:
                print(f"❌ AI routing failed: {e}, using fallback")
        
        # Fallback routing based on keywords
        query_lower = query_text.lower()
        
        if any(keyword in query_lower for keyword in ["soil", "fertilizer", "nutrient", "compost"]):
            return "rishikhet_agent_2"
        elif any(keyword in query_lower for keyword in ["pest", "disease", "insect", "fungus"]):
            return "rishikhet_agent_3"
        elif any(keyword in query_lower for keyword in ["organic", "sustainable", "natural"]):
            return "rishikhet_agent_4"
        elif any(keyword in query_lower for keyword in ["crop", "plant", "harvest", "seed"]):
            return "rishikhet_agent_1"
        else:
            return "farming_agent"

# ============================================================================
# FIXED MAIN AI SYSTEM - THE ORCHESTRATOR
# ============================================================================

class SimpleAgriculturalAI:
    def __init__(self):
        print("🚀 Initializing SimpleAgriculturalAI...")
        
        # Initialize all agents
        try:
            self.rishikhet_agents = [
                RishikhetAgent1(),
                RishikhetAgent2(),
                RishikhetAgent3(),
                RishikhetAgent4(),
                RainForecastAgent()
            ]
            print(f"✅ Initialized {len(self.rishikhet_agents)} Rishikhet agents")
        except Exception as e:
            print(f"❌ Failed to initialize Rishikhet agents: {e}")
            self.rishikhet_agents = []
        
        try:
            self.farming_agent = FarmingAgent()
            print("✅ Initialized FarmingAgent")
        except Exception as e:
            print(f"❌ Failed to initialize FarmingAgent: {e}")
            self.farming_agent = None
        
        try:
            self.web_based_agent = WebBasedAgent()
            print("✅ Initialized WebBasedAgent")
        except Exception as e:
            print(f"❌ Failed to initialize WebBasedAgent: {e}")
            self.web_based_agent = None
        
        # Combine all agents
        self.all_agents = self.rishikhet_agents.copy()
        if self.farming_agent:
            self.all_agents.append(self.farming_agent)
        if self.web_based_agent:
            self.all_agents.append(self.web_based_agent)
        
        # Initialize the LLM-based router
        try:
            self.router = LLMRouter(self.all_agents)
            print("✅ Initialized LLMRouter")
        except Exception as e:
            print(f"❌ Failed to initialize LLMRouter: {e}")
            self.router = None
        
        print(f"🎉 SimpleAgriculturalAI initialized with {len(self.all_agents)} agents")

    def get_llm_response(self, prompt: str) -> str:
        """Get response from LLM with error handling"""
        try:
            return get_llm_response(prompt)
        except Exception as e:
            print(f"❌ LLM response failed: {e}")
            return f"Error getting LLM response: {str(e)}"

    def ask(self, question: str, farmer_id: Optional[str] = None) -> Dict[str, Any]:
        """
        FIXED: Main method to ask a question to the agricultural AI system.
        """
        print(f"[DEBUG] SimpleAgriculturalAI.ask() called with: '{question}' and farmer_id: {farmer_id}")
        query = Query(text=question, farmer_id=farmer_id)
        chain_of_thought = f"🔍 Starting multi-agent analysis for: '{question}'\n"

        try:
            # Route the query
            if self.router:
                print("[DEBUG] Starting AI router query...")
                best_agent_name = self.router.route_query(query.text)
                print(f"[DEBUG] AI Router completed, selected: '{best_agent_name}'")
                chain_of_thought += f"  - 🤖 AI Router selected '{best_agent_name}'\n"
            else:
                print("[DEBUG] Router not available, using fallback")
                best_agent_name = "farming_agent"
                chain_of_thought += f"  - ⚠️ Router unavailable, defaulting to farming_agent\n"
            
            # Check if this is a farming query that should use cascading approach
            if is_farming_query(question) and best_agent_name not in ["rain_forecast_agent"]:
                print("[DEBUG] Farming query detected - using cascading agent approach")
                chain_of_thought += f"  - 🌾 Farming query detected - using cascading approach\n"
                return self.process_farming_query_cascade(query, chain_of_thought)
            
            # For non-farming queries or weather queries, use single agent approach
            best_agent = next((agent for agent in self.all_agents if agent.name == best_agent_name), self.farming_agent)
            if not best_agent:
                chain_of_thought += f"  - ❌ Agent '{best_agent_name}' not found, using fallback\n"
                return {
                    "answer": "Sorry, I'm having trouble processing your request right now.",
                    "sources": [],
                    "chain_of_thought": chain_of_thought,
                    "agent_used": "error",
                    "success": False
                }
            
            print(f"[DEBUG] Found agent object: {best_agent.name}")
            chain_of_thought += f"  - ✅ Using agent: {best_agent.name}\n"
            
            print(f"[DEBUG] Starting agent processing with {best_agent.name}...")
            final_result = best_agent.process(query)
            print(f"[DEBUG] Agent processing completed. Success: {final_result.success}")
            
            chain_of_thought += f"  - 📝 {best_agent.name.title()} returned response (Success: {final_result.success})\n"
            chain_of_thought += f"  - 🔍 Final answer compiled\n"
            
            return {
                "answer": final_result.response,
                "sources": final_result.sources,
                "chart_path": final_result.chart_path,
                "chain_of_thought": chain_of_thought,
                "agent_used": final_result.agent_name,
                "success": final_result.success
            }
            
        except Exception as e:
            print(f"❌ Error in ask method: {e}")
            chain_of_thought += f"  - ❌ Error occurred: {str(e)}\n"
            return {
                "answer": f"Sorry, I encountered an error: {str(e)}",
                "sources": [],
                "chain_of_thought": chain_of_thought,
                "agent_used": "error",
                "success": False
            }

    def process_farming_query_cascade(self, query: Query, chain_of_thought: str) -> Dict[str, Any]:
        """FIXED: Process farming queries using cascading approach"""
        print("[DEBUG] Starting cascading agent processing...")
        chain_of_thought += f"  - 🔄 Starting cascading agent processing\n"
        
        all_responses = []
        all_sources = []
        agents_used = []
        
        try:
            # Step 1: Try Rishikhet agents first
            print("[DEBUG] Step 1: Trying Rishikhet agents...")
            chain_of_thought += f"  - 📚 Step 1: Checking Rishikhet agents\n"
            
            for agent in self.rishikhet_agents:
                if agent.name == "rain_forecast_agent":  # Skip weather agent
                    continue
                    
                try:
                    confidence, matching_topics = agent.can_handle(query)
                    if confidence > 0.5:
                        print(f"[DEBUG] Trying {agent.name} (confidence: {confidence})")
                        result = agent.process(query)
                        if result.success:
                            print(f"[DEBUG] {agent.name} succeeded")
                            all_responses.append(result.response)
                            all_sources.extend(result.sources)
                            agents_used.append(agent.name)
                            chain_of_thought += f"    - ✅ {agent.name}: Success\n"
                            break  # Use first successful Rishikhet agent
                        else:
                            chain_of_thought += f"    - ❌ {agent.name}: Failed\n"
                except Exception as e:
                    print(f"[DEBUG] {agent.name} failed: {e}")
                    chain_of_thought += f"    - ❌ {agent.name}: Error - {str(e)}\n"
            
            # Step 2: Try FarmingAgent
            if self.farming_agent:
                print("[DEBUG] Step 2: Trying FarmingAgent...")
                chain_of_thought += f"  - 🌱 Step 2: Trying FarmingAgent\n"
                try:
                    farming_result = self.farming_agent.process(query)
                    if farming_result.success:
                        all_responses.append(farming_result.response)
                        all_sources.extend(farming_result.sources)
                        agents_used.append(self.farming_agent.name)
                        chain_of_thought += f"    - ✅ FarmingAgent: Success\n"
                    else:
                        chain_of_thought += f"    - ❌ FarmingAgent: Failed\n"
                except Exception as e:
                    print(f"[DEBUG] FarmingAgent failed: {e}")
                    chain_of_thought += f"    - ❌ FarmingAgent: Error - {str(e)}\n"
            
            # Step 3: Try WebBasedAgent
            if self.web_based_agent:
                print("[DEBUG] Step 3: Trying WebBasedAgent...")
                chain_of_thought += f"  - 🌐 Step 3: Trying WebBasedAgent\n"
                try:
                    web_result = self.web_based_agent.process(query)
                    if web_result.success:
                        all_responses.append(web_result.response)
                        all_sources.extend(web_result.sources)
                        agents_used.append(self.web_based_agent.name)
                        chain_of_thought += f"    - ✅ WebBasedAgent: Success\n"
                    else:
                        chain_of_thought += f"    - ❌ WebBasedAgent: Failed\n"
                except Exception as e:
                    print(f"[DEBUG] WebBasedAgent failed: {e}")
                    chain_of_thought += f"    - ❌ WebBasedAgent: Error - {str(e)}\n"
            
            # Synthesize responses
            if all_responses:
                chain_of_thought += f"  - 🔄 Synthesizing {len(all_responses)} responses\n"
                combined_response = self.synthesize_responses(query.text, all_responses)
                chain_of_thought += f"  - ✅ Final synthesis completed\n"
            else:
                combined_response = "I apologize, but I couldn't find a suitable answer to your farming question at the moment."
                chain_of_thought += f"  - ❌ No successful responses to synthesize\n"
            
            return {
                "answer": combined_response,
                "sources": list(set(all_sources)),  # Remove duplicates
                "chain_of_thought": chain_of_thought,
                "agent_used": ", ".join(agents_used) if agents_used else "none",
                "success": len(all_responses) > 0
            }
            
        except Exception as e:
            print(f"❌ Error in cascading process: {e}")
            chain_of_thought += f"  - ❌ Cascading error: {str(e)}\n"
            return {
                "answer": f"Sorry, I encountered an error during processing: {str(e)}",
                "sources": [],
                "chain_of_thought": chain_of_thought,
                "agent_used": "error",
                "success": False
            }

    def synthesize_responses(self, question: str, responses: List[str]) -> str:
        """FIXED: Synthesize multiple agent responses into a coherent answer"""
        if len(responses) == 1:
            return responses[0]
        
        try:
            synthesis_prompt = f"""
            You are an expert agricultural advisor. I have received multiple responses to a farming question from different knowledge sources. Please synthesize these responses into a single, comprehensive, well-formatted, and professional answer.

            Original Question: "{question}"

            Responses to synthesize:
            {chr(10).join([f"Response {i+1}: {resp}" for i, resp in enumerate(responses)])}

            Please provide a synthesized answer that:
            1. Combines the best information from all responses
            2. Is well-structured with clear sections and bullet points
            3. Removes any redundancy or contradictions
            4. Uses professional agricultural terminology
            5. Provides actionable advice
            6. Uses markdown formatting for better readability

            Synthesized Answer:"""
            
            return self.get_llm_response(synthesis_prompt)
            
        except Exception as e:
            print(f"❌ Synthesis failed: {e}")
            return "\n\n".join(responses)  # Fallback: just join responses

# ============================================================================
# DEMO AND CHAT FUNCTIONS
# ============================================================================

def demo():
    """Simple demo function to test the system"""
    print("🌾 Agricultural AI Demo")
    print("=" * 50)
    
    try:
        ai = SimpleAgriculturalAI()
        
        test_questions = [
            "What fertilizer should I use for tomatoes?",
            "How do I control pests in my garden?",
            "What's the weather forecast for Mumbai?"
        ]
        
        for question in test_questions:
            print(f"\n❓ Question: {question}")
            result = ai.ask(question)
            print(f"✅ Answer: {result['answer'][:100]}...")
            print(f"📚 Sources: {len(result['sources'])} sources")
            print(f"🤖 Agent: {result['agent_used']}")
            
    except Exception as e:
        print(f"❌ Demo failed: {e}")

def chat():
    """Interactive chat function"""
    print("🌾 Agricultural AI Chat")
    print("Type 'quit' to exit")
    print("=" * 50)
    
    try:
        ai = SimpleAgriculturalAI()
        
        while True:
            question = input("\n❓ Your question: ").strip()
            if question.lower() in ['quit', 'exit', 'q']:
                break
                
            result = ai.ask(question)
            print(f"\n✅ Answer: {result['answer']}")
            if result['sources']:
                print(f"\n📚 Sources:")
                for source in result['sources']:
                    print(f"  - {source}")
            
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Chat failed: {e}")

if __name__ == "__main__":
    demo()
