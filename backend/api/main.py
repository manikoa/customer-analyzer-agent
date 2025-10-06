"""
FastAPI server for Customer Feedback Analyzer Agent
Provides REST API endpoints for the frontend
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any
import sys
import os
from pathlib import Path

# Add backend to path
backend_path = Path(__file__).parent.parent
sys.path.insert(0, str(backend_path))

# Import backend components
from agent.agent import FeedbackAnalyzerAgent
from config.presets import get_production_config

# Create FastAPI app
app = FastAPI(
    title="Customer Feedback Analyzer API",
    description="AI-powered backend for customer feedback analysis",
    version="1.0.0"
)

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for API
class FeedbackItem(BaseModel):
    id: str
    text: str
    source: str

class AnalysisRequest(BaseModel):
    feedback_items: List[FeedbackItem]

class AnalysisResponse(BaseModel):
    success: bool
    data: Dict[str, Any]
    error: str = None

# Global agent instance
agent = None

@app.on_event("startup")
async def startup_event():
    """Initialize the agent on startup"""
    global agent
    try:
        print("🚀 Initializing Customer Feedback Analyzer Agent...")
        config = get_production_config()
        agent = FeedbackAnalyzerAgent(config=config)
        print("✅ Agent initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize agent: {e}")
        agent = None

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "Customer Feedback Analyzer API",
        "status": "running",
        "agent_ready": agent is not None
    }

@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "agent_initialized": agent is not None,
        "api_version": "1.0.0"
    }

@app.post("/api/analyze", response_model=AnalysisResponse)
async def analyze_feedback(request: AnalysisRequest):
    """
    Analyze customer feedback and return structured results
    """
    if not agent:
        raise HTTPException(
            status_code=503, 
            detail="Agent not initialized. Please check server logs."
        )
    
    try:
        # Extract text from feedback items
        feedback_texts = [item.text for item in request.feedback_items]
        
        if not feedback_texts:
            raise HTTPException(
                status_code=400,
                detail="No feedback text provided"
            )
        
        print(f"📊 Analyzing {len(feedback_texts)} feedback items...")
        
        # Run analysis
        result = agent.analyze_feedback(
            feedback_items=feedback_texts,
            verbose=False
        )
        
        # Convert result to dict format
        response_data = {
            "sentiment_results": [
                s.dict() if s else None 
                for s in result.sentiment_results
            ],
            "category_results": [
                c.dict() if c else None 
                for c in result.category_results
            ],
            "report": result.report.dict() if result.report else None,
            "state": result.state,
            "summary": {
                "total_items": len(feedback_texts),
                "analysis_time": result.state.get('analysis_time'),
                "top_issues": result.get_top_issues(3),
                "critical_alerts": result.get_critical_alerts()
            }
        }
        
        print("✅ Analysis completed successfully")
        
        return AnalysisResponse(
            success=True,
            data=response_data
        )
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        return AnalysisResponse(
            success=False,
            data={},
            error=str(e)
        )

@app.get("/api/stats")
async def get_stats():
    """Get agent statistics"""
    if not agent:
        raise HTTPException(
            status_code=503,
            detail="Agent not initialized"
        )
    
    try:
        stats = agent.get_stats()
        return {
            "success": True,
            "data": stats
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
