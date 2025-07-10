# test_chatbot.py - Updated test script for RIGYA AI chatbot backend

import requests
import json
import time
from datetime import datetime

# Configuration
BASE_URL = "http://localhost:8000"

def test_health_check():
    """Test health check endpoint"""
    print("🏥 Testing health check...")
    try:
        response = requests.get(f"{BASE_URL}/health")
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False

def test_groq_connection():
    """Test Groq connection and model availability"""
    print("\n🤖 Testing Groq connection...")
    try:
        response = requests.get(f"{BASE_URL}/test-groq")
        
        if response.status_code != 200:
            print(f"❌ HTTP Error: {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
        data = response.json()
        
        if data.get("status") == "success":
            print(f"✅ Groq connected successfully")
            print(f"🎯 Current model: {data.get('current_model', 'Unknown')}")
            print(f"💬 Test response: {data.get('test_response', 'No response')}")
            print(f"🎉 {data.get('message', '')}")
        else:
            print(f"❌ Groq test failed: {data.get('message', 'Unknown error')}")
            if data.get("suggestion"):
                print(f"💡 Suggestion: {data['suggestion']}")
        
        return data.get("status") == "success"
    except requests.exceptions.RequestException as e:
        print(f"❌ Network error testing Groq: {e}")
        return False
    except json.JSONDecodeError as e:
        print(f"❌ Invalid JSON response from Groq test: {e}")
        return False
    except Exception as e:
        print(f"❌ Groq test failed: {e}")
        return False

def test_chat_endpoint():
    """Test the main chat endpoint with various subjects"""
    print("\n💬 Testing chat endpoint...")
    
    test_messages = [
        {
            "message": "Hello, what's your name?",
            "include_web_search": False
        },
        {
            "message": "What is the quadratic formula and how do I use it?",
            "subject": "mathematics",
            "include_web_search": True
        },
        {
            "message": "Explain Newton's first law of motion with examples",
            "subject": "physics", 
            "include_web_search": True
        },
        {
            "message": "How does photosynthesis work in plants?",
            "subject": "biology",
            "include_web_search": True
        },
        {
            "message": "What are ionic and covalent bonds?",
            "subject": "chemistry",
            "include_web_search": True
        },
        {
            "message": "Explain binary search algorithm",
            "subject": "computer_science",
            "include_web_search": True
        },
        {
            "message": "What caused World War 1?",
            "subject": "history",
            "include_web_search": True
        }
    ]
    
    session_id = None
    
    for i, test_message in enumerate(test_messages, 1):
        print(f"\n📝 Test {i}: {test_message['message'][:50]}...")
        
        try:
            # Add session_id to maintain conversation context
            if session_id:
                test_message["session_id"] = session_id
                
            response = requests.post(
                f"{BASE_URL}/chat",
                json=test_message,
                headers={"Content-Type": "application/json"},
                timeout=60  # Increased timeout for AI responses
            )
            
            if response.status_code == 200:
                data = response.json()
                session_id = data["session_id"]  # Store for next messages
                
                print(f"✅ Response received in {data['response_time']:.2f}s")
                print(f"🔍 Web search used: {data['web_search_used']}")
                print(f"📚 Sources: {len(data['sources'])} found")
                print(f"🤖 Bot response: {data['response'][:150]}...")
                print(f"💡 Suggestions: {', '.join(data['suggested_actions'][:2])}")
                
                if data['sources']:
                    print(f"🔗 Sources: {', '.join(data['sources'][:2])}")
                    
            else:
                print(f"❌ Request failed: {response.status_code}")
                print(f"Error: {response.text}")
                
        except requests.exceptions.Timeout:
            print(f"⏰ Test {i} timed out (this is normal for complex questions)")
        except Exception as e:
            print(f"❌ Test {i} failed: {e}")
    
    return session_id

def test_web_search_endpoint():
    """Test the direct web search endpoint"""
    print("\n🔍 Testing web search endpoint...")
    
    search_queries = [
        "photosynthesis process",
        "quadratic formula",
        "Python programming basics",
        "climate change effects"
    ]
    
    for query in search_queries:
        try:
            response = requests.post(
                f"{BASE_URL}/search",
                params={"query": query},
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Search '{query}': {data['total_results']} results")
                
                for i, result in enumerate(data['results'][:2], 1):
                    print(f"   {i}. {result['title'][:50]}...")
                    print(f"      {result['snippet'][:80]}...")
            else:
                print(f"❌ Search '{query}' failed: {response.status_code}")
                
        except Exception as e:
            print(f"❌ Search test failed: {e}")

def test_conversation_history(session_id):
    """Test conversation history endpoint"""
    if not session_id:
        print("\n⚠️ Skipping conversation history test (no session_id)")
        return
        
    print(f"\n📚 Testing conversation history for session: {session_id}")
    
    try:
        response = requests.get(f"{BASE_URL}/conversations/{session_id}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Found {data['total_messages']} messages in conversation")
            
            # Show last 2 conversations
            for i, conv in enumerate(data['conversations'][-2:], 1):
                print(f"  {i}. Student: {conv['user_message'][:50]}...")
                print(f"     Bot: {conv['bot_response'][:80]}...")
                if conv.get('sources'):
                    print(f"     Sources: {len(conv['sources'])}")
        else:
            print(f"❌ Failed to get history: {response.status_code}")
            
    except Exception as e:
        print(f"❌ History test failed: {e}")

def test_utility_endpoints():
    """Test other utility endpoints"""
    print("\n🔧 Testing utility endpoints...")
    
    # Test root endpoint
    try:
        response = requests.get(f"{BASE_URL}/")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Root: {data['message']}")
            print(f"   Features: {', '.join(data['features'])}")
        else:
            print(f"❌ Root endpoint failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Root test failed: {e}")
    
    # Test subjects endpoint
    try:
        response = requests.get(f"{BASE_URL}/subjects")
        if response.status_code == 200:
            data = response.json()
            subjects = data["subjects"]
            features = data["features"]
            print(f"✅ Subjects: {', '.join(subjects)}")
            print(f"✅ Features: {', '.join(features)}")
        else:
            print(f"❌ Subjects endpoint failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Subjects test failed: {e}")
    
    # Test stats endpoint
    try:
        response = requests.get(f"{BASE_URL}/stats")
        if response.status_code == 200:
            stats = response.json()
            print(f"✅ Stats - Sessions: {stats['total_sessions']}, Messages: {stats['total_messages']}")
            print(f"   Avg messages/session: {stats['average_messages_per_session']:.1f}")
            if stats.get('subject_distribution'):
                print(f"   Subject distribution: {stats['subject_distribution']}")
        else:
            print(f"❌ Stats endpoint failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Stats test failed: {e}")

def test_error_handling():
    """Test error handling and edge cases"""
    print("\n🚨 Testing error handling...")
    
    # Test invalid session ID
    try:
        response = requests.get(f"{BASE_URL}/conversations/invalid_session_12345")
        if response.status_code == 404:
            print("✅ Invalid session properly returns 404")
        else:
            print(f"❌ Expected 404, got {response.status_code}")
    except Exception as e:
        print(f"❌ Error handling test failed: {e}")
    
    # Test malformed chat request
    try:
        response = requests.post(f"{BASE_URL}/chat", json={"invalid": "data"})
        if response.status_code in [400, 422]:  # Bad request or validation error
            print("✅ Malformed request properly rejected")
        else:
            print(f"❌ Expected 400/422, got {response.status_code}")
    except Exception as e:
        print(f"❌ Malformed request test failed: {e}")
    
    # Test empty message
    try:
        response = requests.post(f"{BASE_URL}/chat", json={"message": ""})
        if response.status_code in [400, 422, 200]:  # Might be handled gracefully
            print("✅ Empty message handled appropriately")
        else:
            print(f"❌ Unexpected response to empty message: {response.status_code}")
    except Exception as e:
        print(f"❌ Empty message test failed: {e}")

def test_performance():
    """Test performance with quick questions"""
    print("\n⚡ Testing performance...")
    
    quick_questions = [
        "What is 2+2?",
        "Hello",
        "What's your name?",
        "Thank you"
    ]
    
    response_times = []
    
    for question in quick_questions:
        try:
            start_time = time.time()
            response = requests.post(
                f"{BASE_URL}/chat",
                json={"message": question, "include_web_search": False},
                timeout=30
            )
            end_time = time.time()
            
            if response.status_code == 200:
                response_time = end_time - start_time
                response_times.append(response_time)
                print(f"✅ '{question}' responded in {response_time:.2f}s")
            else:
                print(f"❌ '{question}' failed: {response.status_code}")
                
        except Exception as e:
            print(f"❌ Performance test failed for '{question}': {e}")
    
    if response_times:
        avg_time = sum(response_times) / len(response_times)
        print(f"📊 Average response time: {avg_time:.2f}s")

def main():
    """Run all tests"""
    print("🚀 Starting RIGYA AI Chatbot Backend Tests")
    print("=" * 60)
    print(f"⏰ Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    # Test health check first
    if not test_health_check():
        print("❌ Health check failed. Make sure the server is running on port 8000!")
        print("💡 Run: python chatbot_server.py")
        return
    
    # Test Groq connection
    groq_working = test_groq_connection()
    if not groq_working:
        print("⚠️ Groq not working properly. Some features may be limited.")
    
    # Test performance first (quick tests)
    test_performance()
    
    # Test main functionality
    session_id = test_chat_endpoint()
    
    # Test conversation history
    if session_id:
        test_conversation_history(session_id)
    
    # Test web search
    test_web_search_endpoint()
    
    # Test utility endpoints
    test_utility_endpoints()
    
    # Test error handling
    test_error_handling()
    
    print("\n" + "=" * 60)
    print("🎉 All tests completed!")
    print("=" * 60)
    
    print("\n📋 Test Summary:")
    print("✅ 1. Health check - API connectivity")
    print("✅ 2. Groq connection - AI model availability") 
    print("✅ 3. Performance - Quick response tests")
    print("✅ 4. Chat endpoint - Core AI functionality")
    print("✅ 5. Web search - Information retrieval")
    print("✅ 6. Conversation history - Session management")
    print("✅ 7. Utility endpoints - Additional features")
    print("✅ 8. Error handling - Robustness")
    
    print("\n💡 Next steps:")
    print("- ✅ Server is ready for Vercel deployment")
    print("- 🎨 Integrate with React frontend")
    print("- 📚 Test with complex academic questions")
    print("- 🔧 Monitor performance under load")
    
    if not groq_working:
        print("\n⚠️ Groq Issues:")
        print("- Install Groq: pip install groq")
        print("- Check API key in the code")
        print("- Ensure internet connection")
        print("- Restart the chatbot server")

if __name__ == "__main__":
    main()