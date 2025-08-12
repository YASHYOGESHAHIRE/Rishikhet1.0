"""
Test script to debug YouTube agent issues.
"""

from agents import YouTubeAgent
from utils import Query
import requests

def test_youtube_agent():
    """Test the YouTube agent functionality and debug issues."""
    print("🧪 Testing YouTube Agent...")
    
    # Initialize the agent
    agent = YouTubeAgent()
    
    print(f"📋 Agent name: {agent.name}")
    print(f"🔑 API key configured: {'Yes' if agent.api_key else 'No'}")
    print(f"🌐 API URL: {agent.api_url}")
    print(f"📝 Topics: {agent.topics}")
    
    # Test query
    test_query = "video tutorial in planting"
    query = Query(text=test_query)
    
    print(f"\n🔍 Testing query: '{test_query}'")
    
    # Test can_handle method
    confidence, topics = agent.can_handle(query)
    print(f"🎯 Can handle confidence: {confidence}")
    print(f"📝 Matching topics: {topics}")
    
    # Test search term processing
    keywords_to_remove = agent.topics + ["video", "youtube", "of", "a", "an", "the"]
    search_term = query.text.lower()
    for keyword in keywords_to_remove:
        search_term = search_term.replace(keyword, "")
    search_term = search_term.strip()
    
    # Add farming context
    farming_keywords = ["farming", "agriculture", "crop", "livestock", "irrigation", "tractor"]
    if not any(keyword in search_term.lower() for keyword in farming_keywords):
        search_term += " farming agriculture"
    
    print(f"🔤 Processed search term: '{search_term}'")
    
    # Test API call manually
    params = {
        'key': agent.api_key,
        'q': search_term,
        'part': 'snippet',
        'type': 'video',
        'maxResults': 5,
        'order': 'relevance',
        'videoDefinition': 'any',
        'videoDuration': 'any'
    }
    
    print(f"\n🌐 Testing YouTube API call...")
    print(f"📡 URL: {agent.api_url}")
    print(f"📋 Params: {params}")
    
    try:
        response = requests.get(agent.api_url, params=params, timeout=10)
        print(f"📊 Response status: {response.status_code}")
        print(f"📄 Response headers: {dict(response.headers)}")
        
        if response.status_code != 200:
            print(f"❌ API Error: {response.status_code}")
            print(f"📄 Response text: {response.text[:500]}")
        else:
            data = response.json()
            print(f"✅ API call successful")
            print(f"📊 Response keys: {list(data.keys())}")
            
            if 'items' in data:
                print(f"🎥 Found {len(data['items'])} videos")
                for i, item in enumerate(data['items'][:2], 1):  # Show first 2
                    print(f"  {i}. {item['snippet']['title']}")
            else:
                print(f"❌ No 'items' in response")
                print(f"📄 Full response: {data}")
                
    except Exception as e:
        print(f"❌ API call failed: {e}")
    
    # Test full agent process
    print(f"\n🔄 Testing full agent process...")
    result = agent.process(query)
    print(f"✅ Success: {result.success}")
    print(f"📝 Response: {result.response[:200]}...")
    print(f"📊 Confidence: {result.confidence}")
    if hasattr(result, 'data') and result.data:
        print(f"📊 Data keys: {list(result.data.keys())}")

if __name__ == "__main__":
    test_youtube_agent()
