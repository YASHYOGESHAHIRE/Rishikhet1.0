"""
Test script to verify ImageAgent integration and functionality.
"""

from routing import SimpleAgriculturalAI
from utils import Query

def test_image_agent():
    """Test the ImageAgent functionality."""
    print("🧪 Testing ImageAgent Integration...")
    
    # Initialize the AI system
    ai_system = SimpleAgriculturalAI()
    
    # Test queries that should trigger ImageAgent
    test_queries = [
        "show me image of solar agriculture",
        "find image of tomato farming",
        "picture of irrigation systems",
        "search for image of organic farming"
    ]
    
    print(f"\n📋 Testing {len(test_queries)} image-related queries:")
    
    for i, query_text in enumerate(test_queries, 1):
        print(f"\n--- Test {i}: '{query_text}' ---")
        
        # Create query object
        query = Query(text=query_text)
        
        # Check if ImageAgent can handle this query
        image_agent = ai_system.image_agent
        confidence, matching_topics = image_agent.can_handle(query)
        
        print(f"🎯 ImageAgent confidence: {confidence:.2f}")
        print(f"📝 Matching topics: {matching_topics}")
        
        if confidence > 0:
            print("✅ ImageAgent should handle this query")
            
            # Test the actual processing (commented out to avoid API calls)
            # result = image_agent.process(query)
            # print(f"🔍 Result: {result.response}")
            # print(f"🖼️ Images found: {len(result.image_urls)}")
        else:
            print("❌ ImageAgent won't handle this query")
    
    # Test system-level routing
    print(f"\n🔄 Testing system-level routing...")
    test_query = "show me image of solar panels in agriculture"
    
    # Check all agents to see which ones can handle the query
    query = Query(text=test_query)
    print(f"\nQuery: '{test_query}'")
    
    for agent in ai_system.all_agents:
        confidence, topics = agent.can_handle(query)
        if confidence > 0:
            print(f"✅ {agent.name}: confidence={confidence:.2f}, topics={topics}")
    
    print(f"\n🎉 ImageAgent integration test complete!")
    print(f"📊 Total agents in system: {len(ai_system.all_agents)}")
    print(f"🖼️ ImageAgent included: {'image_agent' in [agent.name for agent in ai_system.all_agents]}")

if __name__ == "__main__":
    test_image_agent()
