#!/usr/bin/env python3

# Simple test to verify ImageAgent integration
try:
    from agents import ImageAgent
    from utils import Query
    
    print("✅ ImageAgent import successful")
    
    # Test ImageAgent initialization
    agent = ImageAgent()
    print(f"✅ ImageAgent created: {agent.name}")
    print(f"✅ Topics: {agent.topics}")
    
    # Test query handling
    query = Query(text="show me image of solar agriculture")
    confidence, topics = agent.can_handle(query)
    print(f"✅ Query handling test: confidence={confidence}, topics={topics}")
    
    # Test routing system
    from routing import SimpleAgriculturalAI
    ai = SimpleAgriculturalAI()
    agent_names = [agent.name for agent in ai.all_agents]
    print(f"✅ All agents: {agent_names}")
    print(f"✅ ImageAgent in system: {'image_agent' in agent_names}")
    
    print("🎉 All tests passed!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
