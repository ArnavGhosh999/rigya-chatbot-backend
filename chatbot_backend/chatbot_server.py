# Cell 1: Imports and Configuration
import asyncio
import json
import logging
import time
import requests
from datetime import datetime
from typing import List, Dict, Optional, Any
import uuid
import re
from urllib.parse import quote_plus

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Try to import groq
try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
GROQ_API_KEY = "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"
GROQ_MODEL = "llama3-70b-8192"  # Fast Llama 3 70B model
# No additional API keys needed - using free APIs (DuckDuckGo, Wikipedia)

# Cell 2: Data Models
class ChatMessage(BaseModel):
    message: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    subject: Optional[str] = None
    include_web_search: bool = True

class ChatResponse(BaseModel):
    response: str
    session_id: str
    timestamp: str
    response_time: float
    sources: List[str] = []
    suggested_actions: List[str] = []
    web_search_used: bool = False

class ConversationHistory(BaseModel):
    user_message: str
    bot_response: str
    timestamp: str
    subject: Optional[str] = None
    sources: List[str] = []

class SearchResult(BaseModel):
    title: str
    url: str
    snippet: str
    relevance_score: float = 0.0

# Cell 3: Web Search Engine Class
class WebSearchEngine:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
    
    async def search_web(self, query: str, max_results: int = 5) -> List[SearchResult]:
        """Search the web for information"""
        try:
            # Use DuckDuckGo Instant Answer API (free)
            search_results = await self._search_duckduckgo(query, max_results)
            
            # If no results, try Wikipedia
            if not search_results:
                search_results = await self._search_wikipedia(query)
            
            return search_results
            
        except Exception as e:
            logger.error(f"Web search failed: {e}")
            return []
    
    async def _search_duckduckgo(self, query: str, max_results: int) -> List[SearchResult]:
        """Search using DuckDuckGo API"""
        try:
            # DuckDuckGo Instant Answer API
            url = f"https://api.duckduckgo.com/?q={quote_plus(query)}&format=json&no_html=1&skip_disambig=1"
            
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            results = []
            
            # Extract abstract if available
            if data.get('Abstract'):
                results.append(SearchResult(
                    title=data.get('AbstractSource', 'DuckDuckGo'),
                    url=data.get('AbstractURL', ''),
                    snippet=data.get('Abstract', ''),
                    relevance_score=0.9
                ))
            
            # Extract related topics
            for topic in data.get('RelatedTopics', [])[:3]:
                if isinstance(topic, dict) and topic.get('Text'):
                    results.append(SearchResult(
                        title=topic.get('FirstURL', '').split('/')[-1] if topic.get('FirstURL') else 'Related',
                        url=topic.get('FirstURL', ''),
                        snippet=topic.get('Text', ''),
                        relevance_score=0.7
                    ))
            
            return results[:max_results]
            
        except Exception as e:
            logger.error(f"DuckDuckGo search failed: {e}")
            return []
    
    async def _search_wikipedia(self, query: str) -> List[SearchResult]:
        """Search Wikipedia as fallback"""
        try:
            # Wikipedia API
            search_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{quote_plus(query)}"
            
            response = self.session.get(search_url, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            if data.get('extract'):
                return [SearchResult(
                    title=data.get('title', 'Wikipedia'),
                    url=data.get('content_urls', {}).get('desktop', {}).get('page', ''),
                    snippet=data.get('extract', ''),
                    relevance_score=0.8
                )]
            
            return []
            
        except Exception as e:
            logger.error(f"Wikipedia search failed: {e}")
            return []
    
    def extract_key_info(self, search_results: List[SearchResult]) -> str:
        """Extract key information from search results"""
        if not search_results:
            return ""
        
        # Combine snippets from top results
        combined_info = ""
        for result in search_results[:3]:  # Use top 3 results
            if result.snippet:
                combined_info += f"{result.snippet}\n\n"
        
        return combined_info.strip()
    
# Cell 4: Main Chatbot Class
class ImprovedStudentChatbot:
    def __init__(self):
        self.web_search = WebSearchEngine()
        self.model_name = GROQ_MODEL
        self.groq_client = None
        self.system_prompt = """You are RIGYA AI, an intelligent student assistant. Your role is to help students learn and understand academic concepts.

Guidelines:
1. Always provide accurate, educational responses
2. Use the provided web search information to give current, factual answers
3. Break down complex topics into understandable parts
4. Provide step-by-step explanations when appropriate
5. If you don't know something, search for information or admit uncertainty
6. Always cite your sources when using web search information
7. Be encouraging and supportive in your responses

When answering questions:
- For math: Show step-by-step solutions
- For science: Explain concepts clearly with examples
- For general topics: Provide comprehensive, well-structured answers
- Always make learning engaging and accessible"""
        
        # Initialize Groq client
        self._setup_groq_client()
    
    def _setup_groq_client(self):
        """Initialize Groq client"""
        if not GROQ_AVAILABLE:
            logger.warning("Groq not available. Install with: pip install groq")
            return False
        
        try:
            self.groq_client = Groq(api_key=GROQ_API_KEY)
            logger.info(f"✅ Groq client initialized successfully")
            logger.info(f"🤖 Using model: {self.model_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error setting up Groq client: {e}")
            return False
    
    def analyze_subject(self, message: str) -> str:
        """Analyze the subject area of the question"""
        message_lower = message.lower()
        
        subject_keywords = {
            "mathematics": ["math", "algebra", "geometry", "calculus", "trigonometry", "statistics", 
                          "equation", "formula", "solve", "derivative", "integral", "graph", "function"],
            "physics": ["physics", "force", "energy", "motion", "velocity", "acceleration", "gravity",
                       "electricity", "magnetism", "wave", "quantum", "thermodynamics", "mechanics"],
            "chemistry": ["chemistry", "element", "compound", "reaction", "bond", "atom", "molecule",
                         "periodic", "chemical", "organic", "inorganic", "stoichiometry", "acid", "base"],
            "biology": ["biology", "cell", "dna", "gene", "evolution", "organism", "anatomy", "physiology",
                       "ecology", "photosynthesis", "respiration", "genetics", "protein", "enzyme"],
            "computer_science": ["programming", "algorithm", "code", "software", "computer", "data structure",
                               "python", "java", "javascript", "database", "ai", "machine learning"],
            "history": ["history", "historical", "war", "civilization", "ancient", "medieval", "modern",
                       "revolution", "empire", "dynasty", "culture", "society"],
            "literature": ["literature", "novel", "poem", "poetry", "author", "book", "story", "essay",
                         "shakespeare", "writing", "literary", "character", "plot", "theme"]
        }
        
        for subject, keywords in subject_keywords.items():
            if any(keyword in message_lower for keyword in keywords):
                return subject
        
        return "general"
    
    async def generate_response(self, message: str, context: List[ConversationHistory] = None, 
                              use_web_search: bool = True) -> tuple[str, List[str], bool]:
        """Generate response using Groq AI and web search"""
        sources = []
        web_search_used = False
        
        try:
            # Check if this is a greeting
            if self._is_greeting(message):
                return self._get_greeting_response(), [], False
            
            # Perform web search for factual information
            search_results = []
            if use_web_search:
                search_results = await self.web_search.search_web(message)
                if search_results:
                    web_search_used = True
                    sources = [result.url for result in search_results if result.url]
            
            # Generate response using Groq
            response = await self._generate_groq_response(message, search_results, context)
            
            return response, sources, web_search_used
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return "I apologize, but I'm having trouble processing your request right now. Please try again!", [], False
    
    def _is_greeting(self, message: str) -> bool:
        """Check if message is a greeting"""
        greetings = ["hello", "hi", "hey", "good morning", "good afternoon", "good evening", "what's up"]
        return any(greeting in message.lower() for greeting in greetings)
    
    def _get_greeting_response(self) -> str:
        """Get appropriate greeting response"""
        return "Hello! I'm RIGYA AI, your intelligent study assistant powered by Groq's lightning-fast AI. I can help you with academic questions across various subjects including mathematics, physics, chemistry, biology, computer science, and more. What would you like to learn about today?"
    
    async def _generate_groq_response(self, message: str, search_results: List[SearchResult], 
                                     context: List[ConversationHistory] = None) -> str:
        """Generate response using Groq AI"""
        try:
            if not GROQ_AVAILABLE or not self.groq_client:
                return "I'm currently unable to access my AI capabilities. Please ensure Groq is properly configured."
            
            # Build context for the AI
            context_text = self._build_context(message, search_results, context)
            
            # Generate response using Groq
            chat_completion = self.groq_client.chat.completions.create(
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": context_text}
                ],
                model=self.model_name,
                temperature=0.7,
                max_tokens=1000,
                top_p=0.9,
                stream=False
            )
            
            return chat_completion.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"Groq response generation failed: {e}")
            
            # Fallback response using search results
            if search_results:
                return self._create_fallback_response(message, search_results)
            
            return "I'm having trouble generating a response right now. Please ensure you have a stable internet connection and try again."
    
    def _build_context(self, message: str, search_results: List[SearchResult], 
                      context: List[ConversationHistory] = None) -> str:
        """Build context for the AI response"""
        context_parts = []
        
        # Add conversation history
        if context:
            context_parts.append("Previous conversation:")
            for conv in context[-3:]:  # Last 3 exchanges
                context_parts.append(f"User: {conv.user_message}")
                context_parts.append(f"Assistant: {conv.bot_response}")
        
        # Add web search results
        if search_results:
            context_parts.append("\nWeb search information:")
            for result in search_results:
                context_parts.append(f"Source: {result.title}")
                context_parts.append(f"Content: {result.snippet}")
                context_parts.append("---")
        
        # Add current question
        context_parts.append(f"\nCurrent question: {message}")
        context_parts.append("\nPlease provide a comprehensive, educational response based on the above information. If using web search information, please cite your sources.")
        
        return "\n".join(context_parts)
    
    def _create_fallback_response(self, message: str, search_results: List[SearchResult]) -> str:
        """Create a fallback response using search results"""
        if not search_results:
            return "I don't have enough information to answer that question right now."
        
        response_parts = []
        response_parts.append("Based on my search, here's what I found:")
        
        for result in search_results[:2]:  # Top 2 results
            if result.snippet:
                response_parts.append(f"\n{result.snippet}")
        
        response_parts.append(f"\nSources: {', '.join([r.title for r in search_results if r.title])}")
        
        return "\n".join(response_parts)
    
    def generate_suggestions(self, subject: str, message: str) -> List[str]:
        """Generate helpful suggestions based on subject"""
        suggestions_map = {
            "mathematics": [
                "Would you like me to show you step-by-step solutions?",
                "Need help with practice problems?",
                "Want to see visual representations or graphs?"
            ],
            "physics": [
                "Would you like real-world examples?",
                "Need help with the mathematical derivations?",
                "Want to see how this connects to other physics concepts?"
            ],
            "chemistry": [
                "Would you like to see molecular structures?",
                "Need help balancing chemical equations?",
                "Want to explore laboratory applications?"
            ],
            "biology": [
                "Would you like diagrams or visual explanations?",
                "Need help connecting this to other biological systems?",
                "Want to see how this relates to human health?"
            ],
            "computer_science": [
                "Would you like to see code examples?",
                "Need help with debugging or optimization?",
                "Want to explore practical applications?"
            ],
            "general": [
                "Would you like more detailed explanations?",
                "Need help with related topics?",
                "Want study tips for this subject?"
            ]
        }
        
        return suggestions_map.get(subject, suggestions_map["general"])
    
# Cell 5: FastAPI Application Setup
# Initialize FastAPI app
app = FastAPI(
    title="RIGYA AI Student Chatbot",
    description="Intelligent AI-powered chatbot for student assistance with web search capabilities",
    version="3.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory storage (use database in production)
conversations: Dict[str, List[ConversationHistory]] = {}
active_sessions: Dict[str, dict] = {}

# Initialize chatbot
chatbot = ImprovedStudentChatbot()

# Startup function (new FastAPI way)
async def startup_event():
    """Initialize chatbot on startup"""
    logger.info("Starting RIGYA AI Student Chatbot...")
    logger.info(f"Groq available: {GROQ_AVAILABLE}")
    if GROQ_AVAILABLE:
        logger.info(f"🚀 Using Groq AI with model: {GROQ_MODEL}")
        logger.info("⚡ Lightning-fast responses enabled!")
    else:
        logger.warning("⚠️ Groq not available. Install with: pip install groq")

# Register startup event
app.add_event_handler("startup", startup_event)

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "RIGYA AI Student Chatbot v3.0 - Powered by Groq",
        "status": "active",
        "features": ["Web Search", "Groq Lightning AI", "Multi-subject Support"],
        "groq_available": GROQ_AVAILABLE,
        "model": GROQ_MODEL
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "version": "3.0",
        "groq_available": GROQ_AVAILABLE,
        "web_search": "enabled",
        "model": GROQ_MODEL
    }

# Cell 6: API Endpoints
@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(message_data: ChatMessage):
    """Main chat endpoint with web search and Groq AI"""
    start_time = time.time()
    
    try:
        # Generate session ID if not provided
        session_id = message_data.session_id or str(uuid.uuid4())
        
        # Get conversation history
        history = conversations.get(session_id, [])
        
        # Generate response with web search
        bot_response, sources, web_search_used = await chatbot.generate_response(
            message_data.message, 
            history, 
            message_data.include_web_search
        )
        
        # Analyze subject
        subject = chatbot.analyze_subject(message_data.message)
        
        # Calculate response time
        response_time = time.time() - start_time
        
        # Generate suggestions
        suggestions = chatbot.generate_suggestions(subject, message_data.message)
        
        # Store conversation
        conversation_entry = ConversationHistory(
            user_message=message_data.message,
            bot_response=bot_response,
            timestamp=datetime.now().isoformat(),
            subject=subject,
            sources=sources
        )
        
        if session_id not in conversations:
            conversations[session_id] = []
        conversations[session_id].append(conversation_entry)
        
        # Update active session
        active_sessions[session_id] = {
            "last_activity": datetime.now().isoformat(),
            "user_id": message_data.user_id,
            "message_count": len(conversations[session_id]),
            "subject": subject
        }
        
        return ChatResponse(
            response=bot_response,
            session_id=session_id,
            timestamp=datetime.now().isoformat(),
            response_time=response_time,
            sources=sources,
            suggested_actions=suggestions,
            web_search_used=web_search_used
        )
        
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/conversations/{session_id}")
async def get_conversation_history(session_id: str):
    """Get conversation history for a session"""
    if session_id not in conversations:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return {
        "session_id": session_id,
        "conversations": conversations[session_id],
        "total_messages": len(conversations[session_id]),
        "session_info": active_sessions.get(session_id, {})
    }

@app.delete("/conversations/{session_id}")
async def clear_conversation(session_id: str):
    """Clear conversation history for a session"""
    if session_id in conversations:
        del conversations[session_id]
    if session_id in active_sessions:
        del active_sessions[session_id]
    
    return {"message": "Conversation cleared successfully"}

@app.get("/subjects")
async def get_supported_subjects():
    """Get list of supported subjects"""
    return {
        "subjects": [
            "Mathematics", "Physics", "Chemistry", "Biology", 
            "Computer Science", "History", "Literature", "General"
        ],
        "features": [
            "Web search integration",
            "Groq Lightning AI",
            "Step-by-step explanations",
            "Source citations",
            "Multi-subject support"
        ]
    }

@app.get("/stats")
async def get_stats():
    """Get chatbot usage statistics"""
    total_conversations = len(conversations)
    total_messages = sum(len(conv) for conv in conversations.values())
    
    # Subject distribution
    subject_counts = {}
    for conv_list in conversations.values():
        for conv in conv_list:
            subject = conv.subject or "unknown"
            subject_counts[subject] = subject_counts.get(subject, 0) + 1
    
    return {
        "total_sessions": total_conversations,
        "total_messages": total_messages,
        "active_sessions": len(active_sessions),
        "average_messages_per_session": total_messages / total_conversations if total_conversations > 0 else 0,
        "subject_distribution": subject_counts,
        "version": "3.0 - Web Search + Groq AI",
        "groq_status": GROQ_AVAILABLE
    }

@app.post("/search")
async def search_endpoint(query: str):
    """Direct web search endpoint"""
    try:
        search_results = await chatbot.web_search.search_web(query)
        return {
            "query": query,
            "results": [
                {
                    "title": result.title,
                    "url": result.url,
                    "snippet": result.snippet,
                    "relevance": result.relevance_score
                }
                for result in search_results
            ],
            "total_results": len(search_results)
        }
    except Exception as e:
        logger.error(f"Search endpoint error: {e}")
        raise HTTPException(status_code=500, detail="Search failed")

@app.get("/test-groq")
async def test_groq():
    """Test Groq connection and model availability"""
    if not GROQ_AVAILABLE:
        return {"status": "error", "message": "Groq not installed"}
    
    try:
        # Test a simple completion
        if chatbot.groq_client:
            test_response = chatbot.groq_client.chat.completions.create(
                messages=[
                    {"role": "user", "content": "Hello, please respond with 'Groq test successful'"}
                ],
                model=GROQ_MODEL,
                max_tokens=50
            )
            
            return {
                "status": "success",
                "current_model": GROQ_MODEL,
                "test_response": test_response.choices[0].message.content,
                "message": "Groq AI is working perfectly!"
            }
        else:
            return {
                "status": "error", 
                "message": "Groq client not initialized",
                "suggestion": "Check your API key"
            }
            
    except Exception as e:
        return {
            "status": "error",
            "message": f"Groq test failed: {str(e)}",
            "suggestion": "Check your Groq API key and internet connection"
        }
    
#Cell 7 : main app.
if __name__ == "__main__":
    import uvicorn
    import os
    
    # Get port from environment (Railway sets this)
    port = int(os.environ.get("PORT", 8000))
    
    # Print startup information
    print("=" * 60)
    print("RIGYA AI Student Chatbot v3.0 - Powered by Groq")
    print("=" * 60)
    print(f"Groq Available: {GROQ_AVAILABLE}")
    print(f"Model: {GROQ_MODEL}")
    print("Web Search: Enabled")
    print(f"Starting on port: {port}")
    print("=" * 60)
    
    # Run the server
    uvicorn.run(
        "chatbot_server:app",
        host="0.0.0.0",
        port=port,
        reload=False  # Don't reload in production
    )
