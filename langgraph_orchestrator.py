"""
LangGraph-based Agent Orchestration System for Agricultural AI.

This module implements a sophisticated agent collaboration system using LangGraph
to manage multi-agent workflows, dynamic routing, and intelligent collaboration.
"""

import json
from typing import Dict, List, Optional, Any, TypedDict, Annotated
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_groq import ChatGroq

from utils import AgentResult, Query, GROQ_API_KEY
from agents import (
    BaseAgent, RishikhetAgent1, RishikhetAgent2, RishikhetAgent3, RishikhetAgent4,
    WebBasedAgent, FarmingAgent, RainForecastAgent, ImageAgent, YouTubeAgent
)


# ============================================================================
# STATE DEFINITIONS
# ============================================================================

class AgentState(TypedDict):
    """State for the agent collaboration workflow."""
    messages: Annotated[List[BaseMessage], add_messages]
    query: str
    original_query: str
    farmer_id: Optional[str]
    selected_agents: List[str]
    agent_results: Dict[str, AgentResult]
    collaboration_history: List[Dict[str, Any]]
    final_result: Optional[AgentResult]
    routing_decision: Optional[str]
    needs_collaboration: bool
    iteration_count: int
    max_iterations: int


# ============================================================================
# LANGGRAPH ORCHESTRATOR
# ============================================================================

class LangGraphOrchestrator:
    """
    LangGraph-based orchestrator for intelligent agent collaboration and routing.
    """
    
    def __init__(self):
        """Initialize the orchestrator with all agents and LangGraph workflow."""
        # Initialize all agents
        self.agents = {
            'rishikhet_1': RishikhetAgent1(),
            'rishikhet_2': RishikhetAgent2(),
            'rishikhet_3': RishikhetAgent3(),
            'rishikhet_4': RishikhetAgent4(),
            'web_based': WebBasedAgent(),
            'farming': FarmingAgent(),
            'rain_forecast': RainForecastAgent(),
            'image': ImageAgent(),
            'youtube': YouTubeAgent()
        }
        
        # Initialize LLM for routing decisions
        self.llm = ChatGroq(
            api_key=GROQ_API_KEY, 
            model="llama3-70b-8192", 
            temperature=0.1, 
            timeout=15
        )
        
        # Build the LangGraph workflow
        self.workflow = self._build_workflow()
        self.app = self.workflow.compile()
    
    def _build_workflow(self) -> StateGraph:
        """Build the LangGraph workflow for agent collaboration."""
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("analyze_query", self._analyze_query)
        workflow.add_node("route_to_agents", self._route_to_agents)
        workflow.add_node("execute_agents", self._execute_agents)
        workflow.add_node("evaluate_results", self._evaluate_results)
        workflow.add_node("collaborate", self._collaborate)
        workflow.add_node("synthesize_final", self._synthesize_final)
        
        # Define the workflow edges
        workflow.set_entry_point("analyze_query")
        
        workflow.add_edge("analyze_query", "route_to_agents")
        workflow.add_edge("route_to_agents", "execute_agents")
        workflow.add_edge("execute_agents", "evaluate_results")
        
        # Conditional routing from evaluate_results
        workflow.add_conditional_edges(
            "evaluate_results",
            self._should_collaborate,
            {
                "collaborate": "collaborate",
                "synthesize": "synthesize_final"
            }
        )
        
        workflow.add_edge("collaborate", "execute_agents")
        workflow.add_edge("synthesize_final", END)
        
        return workflow
    
    def _analyze_query(self, state: AgentState) -> AgentState:
        """Analyze the incoming query to understand intent and requirements."""
        query = state["query"]
        
        analysis_prompt = f"""
        Analyze this agricultural query and provide insights for agent routing:
        
        Query: "{query}"
        
        Provide analysis in JSON format:
        {{
            "query_type": "weather|farming|image|video|general|technical",
            "complexity": "simple|medium|complex",
            "requires_collaboration": true/false,
            "primary_domain": "crop_production|livestock|business|resources|weather|visual",
            "suggested_agents": ["agent1", "agent2", ...],
            "reasoning": "explanation of routing decision"
        }}
        
        Available agents:
        - rishikhet_1: Crop production, planting, harvesting
        - rishikhet_2: Livestock management, animal care
        - rishikhet_3: Business, finance, marketing
        - rishikhet_4: Resources, equipment, technology
        - web_based: Web search for current information
        - farming: General farming knowledge
        - rain_forecast: Weather and rainfall predictions
        - image: Image search and visual content
        - youtube: Video tutorials and demonstrations
        """
        
        try:
            response = self.llm.invoke([HumanMessage(content=analysis_prompt)])
            analysis = json.loads(response.content.strip())
            
            state["routing_decision"] = analysis.get("reasoning", "")
            state["needs_collaboration"] = analysis.get("requires_collaboration", False)
            state["selected_agents"] = analysis.get("suggested_agents", [])
            
            # Add analysis to messages
            state["messages"].append(AIMessage(
                content=f"Query analyzed: {analysis.get('query_type', 'unknown')} complexity, "
                       f"primary domain: {analysis.get('primary_domain', 'general')}"
            ))
            
        except Exception as e:
            # Fallback to simple analysis
            state["selected_agents"] = ["farming", "web_based"]
            state["needs_collaboration"] = True
            state["messages"].append(AIMessage(
                content=f"Fallback analysis used due to error: {str(e)}"
            ))
        
        return state
    
    def _route_to_agents(self, state: AgentState) -> AgentState:
        """Route the query to appropriate agents based on analysis."""
        query = state["query"]
        selected_agents = state["selected_agents"]
        
        # If no agents selected, use confidence-based selection
        if not selected_agents:
            query_obj = Query(text=query)
            agent_scores = []
            
            for agent_name, agent in self.agents.items():
                confidence, topics = agent.can_handle(query_obj)
                if confidence > 0.1:  # Threshold for consideration
                    agent_scores.append((agent_name, confidence, topics))
            
            # Sort by confidence and select top agents
            agent_scores.sort(key=lambda x: x[1], reverse=True)
            selected_agents = [agent[0] for agent in agent_scores[:3]]
        
        # Ensure we have at least one agent
        if not selected_agents:
            selected_agents = ["farming"]
        
        state["selected_agents"] = selected_agents
        state["messages"].append(AIMessage(
            content=f"Routing to agents: {', '.join(selected_agents)}"
        ))
        
        return state
    
    def _execute_agents(self, state: AgentState) -> AgentState:
        """Execute the selected agents and collect their results."""
        query = state["query"]
        selected_agents = state["selected_agents"]
        farmer_id = state.get("farmer_id")
        query_obj = Query(text=query, farmer_id=farmer_id)
        
        # Execute agents in parallel (simulated)
        for agent_name in selected_agents:
            if agent_name in self.agents and agent_name not in state["agent_results"]:
                try:
                    agent = self.agents[agent_name]
                    result = agent.process(query_obj)
                    state["agent_results"][agent_name] = result
                    
                    state["messages"].append(AIMessage(
                        content=f"Agent {agent_name} executed successfully"
                    ))
                    
                except Exception as e:
                    state["messages"].append(AIMessage(
                        content=f"Agent {agent_name} failed: {str(e)}"
                    ))
        
        return state
    
    def _evaluate_results(self, state: AgentState) -> AgentState:
        """Evaluate agent results and determine if collaboration is needed."""
        agent_results = state["agent_results"]
        
        # Simple evaluation: check if we have sufficient results
        total_confidence = sum(
            result.confidence for result in agent_results.values()
        )
        
        # If confidence is low or we explicitly need collaboration
        if total_confidence < 0.7 or state["needs_collaboration"]:
            state["needs_collaboration"] = True
        else:
            state["needs_collaboration"] = False
        
        state["messages"].append(AIMessage(
            content=f"Results evaluated. Total confidence: {total_confidence:.2f}, "
                   f"Collaboration needed: {state['needs_collaboration']}"
        ))
        
        return state
    
    def _should_collaborate(self, state: AgentState) -> str:
        """Decide whether to collaborate or synthesize final result."""
        if (state["needs_collaboration"] and 
            state["iteration_count"] < state["max_iterations"]):
            return "collaborate"
        return "synthesize"
    
    def _collaborate(self, state: AgentState) -> AgentState:
        """Enable collaboration between agents by sharing results and refining queries."""
        state["iteration_count"] += 1
        
        # Analyze current results to determine collaboration strategy
        current_results = state["agent_results"]
        
        collaboration_prompt = f"""
        Current query: "{state['query']}"
        
        Agent results so far:
        {self._format_results_for_collaboration(current_results)}
        
        Based on these results, suggest:
        1. Which additional agents should be consulted?
        2. What specific aspects need more information?
        3. How should the query be refined for better results?
        
        Provide response in JSON format:
        {{
            "additional_agents": ["agent1", "agent2"],
            "refined_query": "refined version of the query",
            "collaboration_reason": "explanation of why collaboration is needed"
        }}
        """
        
        try:
            response = self.llm.invoke([HumanMessage(content=collaboration_prompt)])
            collaboration_plan = json.loads(response.content.strip())
            
            # Add new agents to selection
            additional_agents = collaboration_plan.get("additional_agents", [])
            for agent in additional_agents:
                if agent not in state["selected_agents"]:
                    state["selected_agents"].append(agent)
            
            # Refine query if suggested
            refined_query = collaboration_plan.get("refined_query")
            if refined_query and refined_query != state["query"]:
                state["query"] = refined_query
            
            # Record collaboration history
            state["collaboration_history"].append({
                "iteration": state["iteration_count"],
                "reason": collaboration_plan.get("collaboration_reason", ""),
                "additional_agents": additional_agents,
                "refined_query": refined_query
            })
            
            state["messages"].append(AIMessage(
                content=f"Collaboration iteration {state['iteration_count']}: "
                       f"Adding agents {additional_agents}, refined query"
            ))
            
        except Exception as e:
            # Simple fallback collaboration
            if "web_based" not in state["selected_agents"]:
                state["selected_agents"].append("web_based")
            
            state["messages"].append(AIMessage(
                content=f"Fallback collaboration: added web_based agent due to error: {str(e)}"
            ))
        
        return state
    
    def _synthesize_final(self, state: AgentState) -> AgentState:
        """Synthesize final result from all agent outputs."""
        agent_results = state["agent_results"]
        original_query = state["original_query"]
        
        if not agent_results:
            # No results available
            state["final_result"] = AgentResult(
                agent_name="langgraph_orchestrator",
                response="I apologize, but I couldn't generate a response to your query.",
                confidence=0.0,
                sources=[]
            )
            return state
        
        # Combine all agent responses
        combined_responses = []
        all_sources = []
        all_image_urls = []
        all_video_urls = []
        total_confidence = 0
        
        for agent_name, result in agent_results.items():
            if result.response:
                combined_responses.append(f"**{agent_name.replace('_', ' ').title()}**: {result.response}")
            all_sources.extend(result.sources)
            all_image_urls.extend(result.image_urls)
            all_video_urls.extend(result.video_urls)
            total_confidence += result.confidence
        
        # Calculate average confidence
        avg_confidence = total_confidence / len(agent_results) if agent_results else 0
        
        # Create synthesis prompt
        synthesis_prompt = f"""
        Original query: "{original_query}"
        
        Agent responses:
        {chr(10).join(combined_responses)}
        
        Please synthesize these responses into a comprehensive, coherent answer that:
        1. Directly addresses the user's question
        2. Combines insights from all agents
        3. Removes redundancy
        4. Provides practical, actionable information
        5. Maintains accuracy and relevance
        
        Format the response in a clear, helpful manner for farmers and agricultural professionals.
        """
        
        try:
            response = self.llm.invoke([HumanMessage(content=synthesis_prompt)])
            synthesized_response = response.content.strip()
            
            # Create final result
            state["final_result"] = AgentResult(
                agent_name="langgraph_orchestrator",
                response=synthesized_response,
                confidence=min(avg_confidence, 1.0),
                sources=list(set(all_sources)),  # Remove duplicates
                image_urls=list(set(all_image_urls)),
                video_urls=list(set(all_video_urls))
            )
            
        except Exception as e:
            # Fallback: just combine responses
            fallback_response = "\n\n".join(combined_responses)
            state["final_result"] = AgentResult(
                agent_name="langgraph_orchestrator",
                response=fallback_response,
                confidence=avg_confidence,
                sources=all_sources,
                image_urls=all_image_urls,
                video_urls=all_video_urls
            )
        
        state["messages"].append(AIMessage(
            content="Final result synthesized from all agent outputs"
        ))
        
        return state
    
    def _format_results_for_collaboration(self, results: Dict[str, AgentResult]) -> str:
        """Format agent results for collaboration analysis."""
        formatted = []
        for agent_name, result in results.items():
            formatted.append(f"- {agent_name}: {result.response[:200]}... (confidence: {result.confidence:.2f})")
        return "\n".join(formatted)
    
    def process_query(self, query: str, farmer_id: Optional[str] = None) -> AgentResult:
        """
        Process a query using the LangGraph workflow.
        
        Args:
            query: The user's query string
            farmer_id: Optional farmer ID for personalized responses
            
        Returns:
            AgentResult: The final synthesized result
        """
        # Initialize state
        initial_state = AgentState(
            messages=[HumanMessage(content=query)],
            query=query,
            original_query=query,
            farmer_id=farmer_id,
            selected_agents=[],
            agent_results={},
            collaboration_history=[],
            final_result=None,
            routing_decision=None,
            needs_collaboration=False,
            iteration_count=0,
            max_iterations=3
        )
        
        try:
            # Run the workflow
            final_state = self.app.invoke(initial_state)
            
            # Return the final result
            if final_state.get("final_result"):
                return final_state["final_result"]
            else:
                return AgentResult(
                    agent_name="langgraph_orchestrator",
                    response="I encountered an issue processing your query. Please try again.",
                    confidence=0.0,
                    sources=[]
                )
                
        except Exception as e:
            return AgentResult(
                agent_name="langgraph_orchestrator",
                response=f"An error occurred while processing your query: {str(e)}",
                confidence=0.0,
                sources=[]
            )


# ============================================================================
# ENHANCED AGRICULTURAL AI WITH LANGGRAPH
# ============================================================================

class LangGraphAgriculturalAI:
    """
    Enhanced Agricultural AI system using LangGraph for intelligent agent orchestration.
    """
    
    def __init__(self):
        """Initialize the LangGraph-based AI system."""
        self.orchestrator = LangGraphOrchestrator()
    
    def ask(self, question: str, farmer_id: Optional[str] = None) -> AgentResult:
        """
        Ask a question and get a comprehensive answer using LangGraph orchestration.
        
        Args:
            question: The user's question
            farmer_id: Optional farmer ID for personalized responses
            
        Returns:
            AgentResult: Comprehensive result from agent collaboration
        """
        return self.orchestrator.process_query(question, farmer_id=farmer_id)
    
    def get_available_agents(self) -> List[str]:
        """Get list of available agents."""
        return list(self.orchestrator.agents.keys())
    
    def get_agent_info(self, agent_name: str) -> Dict[str, Any]:
        """Get information about a specific agent."""
        if agent_name in self.orchestrator.agents:
            agent = self.orchestrator.agents[agent_name]
            return {
                "name": agent.name,
                "topics": agent.topics,
                "type": type(agent).__name__
            }
        return {}


# ============================================================================
# DEMO FUNCTIONS
# ============================================================================

def demo_langgraph():
    """Demo function to test the LangGraph system."""
    print("🌾 LangGraph Agricultural AI System Demo")
    print("=" * 50)
    
    ai = LangGraphAgriculturalAI()
    
    test_queries = [
        "How do I improve soil fertility for wheat crops?",
        "What's the rainfall forecast for Delhi next week?",
        "Show me images of tomato diseases",
        "Find videos about organic farming techniques"
    ]
    
    for query in test_queries:
        print(f"\n🔍 Query: {query}")
        print("-" * 30)
        
        result = ai.ask(query, farmer_id=None)
        print(f"Agent: {result.agent_name}")
        print(f"Confidence: {result.confidence:.2f}")
        print(f"Response: {result.response[:200]}...")
        if result.sources:
            print(f"Sources: {len(result.sources)} found")
        if result.image_urls:
            print(f"Images: {len(result.image_urls)} found")
        if result.video_urls:
            print(f"Videos: {len(result.video_urls)} found")


def chat_langgraph():
    """Interactive chat function using LangGraph."""
    print("🌾 LangGraph Agricultural AI Chat")
    print("Type 'quit' to exit, 'agents' to see available agents")
    print("=" * 50)
    
    ai = LangGraphAgriculturalAI()
    
    while True:
        try:
            user_input = input("\n🌱 You: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'bye']:
                print("👋 Goodbye! Happy farming!")
                break
            elif user_input.lower() == 'agents':
                agents = ai.get_available_agents()
                print(f"Available agents: {', '.join(agents)}")
                continue
            elif not user_input:
                continue
            
            print("🤖 AI: Processing your query...")
            result = ai.ask(user_input, farmer_id=None)
            
            print(f"\n🤖 AI ({result.agent_name}, confidence: {result.confidence:.2f}):")
            print(result.response)
            
            if result.sources:
                print(f"\n📚 Sources ({len(result.sources)}):")
                for i, source in enumerate(result.sources[:3], 1):
                    print(f"  {i}. {source}")
            
            if result.image_urls:
                print(f"\n🖼️  Images: {len(result.image_urls)} found")
            
            if result.video_urls:
                print(f"\n🎥 Videos: {len(result.video_urls)} found")
                
        except KeyboardInterrupt:
            print("\n👋 Goodbye! Happy farming!")
            break
        except Exception as e:
            print(f"❌ Error: {str(e)}")


if __name__ == "__main__":
    # Run demo
    demo_langgraph()
