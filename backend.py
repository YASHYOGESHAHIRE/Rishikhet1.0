from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Dict, Any, Optional
import os
from dotenv import load_dotenv

# Load environment variables early so agents can read them
load_dotenv()  # loads from .env in project root if present

# Import your existing classes
from routing import SimpleAgriculturalAI, Query  # absolute import
from profile_models import FarmerProfile, ProfileManager, ProfileUpdateRequest

app = FastAPI()

# Mount static folder to serve index.html + assets
app.mount("/static", StaticFiles(directory="static"), name="static")
# Mount character assets for the talking character UI
app.mount("/character", StaticFiles(directory="character"), name="character")

# Instantiate your AI assistant once
ai_assistant = SimpleAgriculturalAI()


# Initialize profile manager
profile_manager = ProfileManager()

# Request model for receiving question
class QuestionRequest(BaseModel):
    text: str
    farmer_id: Optional[str] = None  # Optional farmer ID for personalized responses

# Serve your frontend HTML
@app.get("/", response_class=HTMLResponse)
async def serve_homepage():
    with open("static/index.html", "r", encoding="utf-8") as f:
        return f.read()

# API endpoint to ask question
@app.post("/ask")
async def ask_question(request: QuestionRequest):
    try:
        print(f"Received question: {request.text}")
        print("Starting AI assistant processing...")
        
        # Get farmer profile for personalization if farmer_id provided
        farmer_context = ""
        if request.farmer_id:
            profile = profile_manager.load_profile(request.farmer_id)
            if profile:
                farmer_context = profile.get_context_for_ai()
                print(f"Using farmer context: {farmer_context}")
        
        # Enhance question with farmer context for personalized responses
        enhanced_question = request.text
        if farmer_context:
            enhanced_question = f"Context: {farmer_context}\n\nQuestion: {request.text}"
        
        # Call your existing AI assistant ask method
        result = ai_assistant.ask(enhanced_question)
        
        print("AI assistant processing completed!")
        print(f"AI result keys: {result.keys()}")
        print(f"Success: {result.get('success', 'Unknown')}")
        print(f"Agent used: {result.get('agent_used', 'Unknown')}")
        
        # Ensure all required fields are present
        response_data = {
            "answer": result.get("answer", "No answer provided"),
            "sources": result.get("sources", []),
            "chain_of_thought": result.get("chain_of_thought", "No chain of thought available"),
            "agent_used": result.get("agent_used", "unknown"),
            "success": result.get("success", False),
            "chart_path": result.get("chart_path", None),
            "image_urls": result.get("image_urls", []),  # Include image URLs for direct display
            "agent_details": result.get("agent_details", [])
        }
        
        print(f"Sending response: {len(str(response_data))} characters")
        return JSONResponse(content=response_data)
        
    except Exception as e:
        print(f"Error in ask_question: {str(e)}")
        import traceback
        traceback.print_exc()
        
        error_response = {
            "answer": f"Error processing request: {str(e)}",
            "sources": [],
            "chain_of_thought": f"Error occurred: {str(e)}",
            "agent_used": "error",
            "success": False,
            "chart_path": None,
            "image_urls": []
        }
        return JSONResponse(content=error_response, status_code=500)

# Image analysis endpoint (plant/soil)
@app.post("/analyze-image")
async def analyze_image(file: UploadFile = File(...), mode: str = Form("plant"), demo_mode: str = Form("false")):
    try:
        if not file or not file.filename:
            raise HTTPException(status_code=400, detail="No image file provided")

        contents = await file.read()
        result = ai_assistant.analyze_image(contents, mode=mode.lower().strip() if mode else "plant", demo_mode=demo_mode.lower() == "true")

        response_data = {
            "answer": result.get("answer"),
            "sources": result.get("sources", []),
            "agent_used": result.get("agent_used", "vision_analysis_agent"),
            "success": result.get("success", False),
        }
        return JSONResponse(content=response_data)
    except HTTPException:
        raise
    except Exception as e:
        return JSONResponse(content={
            "answer": f"Error analyzing image: {str(e)}",
            "sources": [],
            "agent_used": "vision_analysis_agent",
            "success": False
        }, status_code=500)

# Profile management endpoints

@app.post("/profile")
async def create_profile(profile: FarmerProfile):
    """Create a new farmer profile"""
    try:
        if profile_manager.save_profile(profile):
            return JSONResponse(content={
                "success": True,
                "message": "Profile created successfully",
                "farmer_id": profile.farmer_id,
                "completeness": profile.calculate_completeness()
            })
        else:
            raise HTTPException(status_code=500, detail="Failed to save profile")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/profile/{farmer_id}")
async def get_profile(farmer_id: str):
    """Get farmer profile by ID"""
    try:
        profile = profile_manager.load_profile(farmer_id)
        if profile:
            return JSONResponse(content=profile.dict())
        else:
            raise HTTPException(status_code=404, detail="Profile not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/profile/{farmer_id}")
async def update_profile(farmer_id: str, updates: Dict[str, Any]):
    """Update farmer profile"""
    try:
        profile = profile_manager.update_profile(farmer_id, updates)
        if profile:
            return JSONResponse(content={
                "success": True,
                "message": "Profile updated successfully",
                "completeness": profile.calculate_completeness(),
                "profile": profile.dict()
            })
        else:
            raise HTTPException(status_code=404, detail="Profile not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/profile/{farmer_id}")
async def delete_profile(farmer_id: str):
    """Delete farmer profile"""
    try:
        if profile_manager.delete_profile(farmer_id):
            return JSONResponse(content={
                "success": True,
                "message": "Profile deleted successfully"
            })
        else:
            raise HTTPException(status_code=500, detail="Failed to delete profile")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/profiles")
async def list_profiles():
    """List all farmer profiles"""
    try:
        profile_ids = profile_manager.list_profiles()
        profiles = []
        for farmer_id in profile_ids:
            profile = profile_manager.load_profile(farmer_id)
            if profile:
                profiles.append({
                    "farmer_id": profile.farmer_id,
                    "name": profile.name,
                    "location": profile.location,
                    "completeness": profile.profile_completeness,
                    "updated_at": profile.updated_at.isoformat()
                })
        return JSONResponse(content={"profiles": profiles})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Serve profile page
@app.get("/profile-page", response_class=HTMLResponse)
async def serve_profile_page():
    """Serve the farmer profile management page"""
    try:
        with open("static/profile.html", "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Profile page not found")

# Serve questions try out page
@app.get("/questions", response_class=HTMLResponse)
async def serve_questions_page():
    """Serve the Questions Try Out page"""
    try:
        with open("static/questions.html", "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Questions page not found")

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting Agricultural AI Backend Server...")
    print("🌐 Server will be available at: http://localhost:8000")
    print("📱 Open your browser and navigate to: http://localhost:8000")
    print("🛑 Press Ctrl+C to stop the server")
    print("="*50)
    
    uvicorn.run(
        "backend:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
