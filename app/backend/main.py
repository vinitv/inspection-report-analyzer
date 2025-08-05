"""
FastAPI Backend for Inspection Report Analyzer
"""

import os
import uuid
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from pathlib import Path

from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn

from app.backend.models import (
    ChatRequest, ChatResponse, ReportAnalysis, 
    UploadResponse, ErrorResponse, RateLimitInfo
)
from app.backend.agent import InspectionAnalyzer
from app.backend.utils import process_pdf, RateLimiter, setup_logging

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="California Property Inspection Analyzer",
    description="AI-powered analysis of property inspection reports with cost estimates and priority recommendations",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8501", "http://127.0.0.1:8501",  # Streamlit
        "http://localhost:3000", "http://127.0.0.1:3000",  # HTML Frontend
        "*"  # Allow all origins for development
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# Initialize components
analyzer = InspectionAnalyzer()
rate_limiter = RateLimiter(max_reports_per_day=3)

# Security
security = HTTPBearer(auto_error=False)

# In-memory session storage (use Redis in production)
sessions: Dict[str, Dict] = {}

@app.on_event("startup")
async def startup_event():
    """Initialize the application on startup"""
    logger.info("🚀 Starting California Property Inspection Analyzer API")
    await analyzer.initialize()
    logger.info("✅ Inspection analyzer initialized successfully")

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0",
        "analyzer_status": "ready" if analyzer.is_ready else "initializing"
    }

def get_session_id(credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)) -> str:
    """Get or create session ID"""
    if credentials and credentials.credentials:
        session_id = credentials.credentials
    else:
        session_id = str(uuid.uuid4())
    
    # Initialize session if it doesn't exist
    if session_id not in sessions:
        sessions[session_id] = {
            "created_at": datetime.now(),
            "reports_uploaded": 0,
            "conversation_history": [],
            "current_report": None
        }
    
    return session_id

@app.post("/api/upload-report", response_model=UploadResponse)
async def upload_inspection_report(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    session_id: str = Depends(get_session_id)
):
    """
    Upload and analyze a property inspection report (PDF)
    """
    try:
        # Check rate limiting
        session = sessions[session_id]
        if not rate_limiter.can_upload(session_id, session["reports_uploaded"]):
            raise HTTPException(
                status_code=429,
                detail="Daily report limit exceeded (3 reports per day)"
            )
        
        # Validate file
        if not file.filename.endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files are allowed")
        
        if file.size > 10 * 1024 * 1024:  # 10MB limit
            raise HTTPException(status_code=400, detail="File size must be less than 10MB")
        
        # Save uploaded file
        upload_dir = Path("app/static/uploads")
        upload_dir.mkdir(exist_ok=True)
        
        file_id = str(uuid.uuid4())
        file_path = upload_dir / f"{file_id}_{file.filename}"
        
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Process PDF in background
        background_tasks.add_task(
            process_pdf_analysis, 
            file_path, file_id, session_id
        )
        
        # Update session
        session["reports_uploaded"] += 1
        session["current_report"] = file_id
        
        logger.info(f"📄 PDF uploaded successfully: {file.filename} (Session: {session_id[:8]})")
        
        return UploadResponse(
            file_id=file_id,
            filename=file.filename,
            status="processing",
            message="PDF uploaded successfully. Analysis in progress...",
            session_id=session_id
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Upload error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

async def process_pdf_analysis(file_path: Path, file_id: str, session_id: str):
    """Background task to process PDF without generating initial analysis"""
    try:
        logger.info(f"🔄 Processing PDF for file {file_id}")
        
        # Extract text from PDF and add to vector store
        pdf_text = process_pdf(file_path)
        
        # Add the PDF to the analyzer's knowledge base
        await analyzer.add_document(pdf_text, file_id)
        
        # Store minimal analysis info in session (no summary generated)
        if session_id in sessions:
            sessions[session_id]["analysis"] = {
                "file_id": file_id,
                "timestamp": datetime.now().isoformat(),
                "summary": "Document processed and ready for questions",
                "strategy_used": "Document Processing",
                "pdf_text": pdf_text,
                "status": "completed"
            }
        
        logger.info(f"✅ PDF processed successfully for file {file_id}")
        
    except Exception as e:
        logger.error(f"❌ PDF processing error for file {file_id}: {str(e)}")
        if session_id in sessions:
            sessions[session_id]["analysis"] = {
                "file_id": file_id,
                "status": "error",
                "error": str(e)
            }

@app.get("/api/report-status/{file_id}")
async def get_report_status(
    file_id: str,
    session_id: str = Depends(get_session_id)
):
    """Check the status of a report analysis"""
    session = sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    analysis = session.get("analysis")
    if not analysis:
        return {"status": "processing", "message": "Analysis in progress..."}
    
    # Log for debugging
    logger.info(f"Status check - Session: {session_id[:8]}, File: {file_id}, Analysis file: {analysis.get('file_id', 'None')}, Status: {analysis.get('status', 'None')}")
    
    # If file_id matches and status is completed
    if analysis.get("file_id") == file_id and analysis.get("status") == "completed":
        return {
            "status": "completed",
            "analysis": {
                "summary": analysis["summary"],
                "strategy_used": analysis["strategy_used"],
                "timestamp": analysis["timestamp"]
            }
        }
    
    # If file_id matches but there's an error
    if analysis.get("file_id") == file_id and analysis.get("status") == "error":
        return {"status": "error", "message": analysis.get("error", "Analysis failed")}
    
    # If there's any completed analysis in the session (regardless of file_id)
    if analysis.get("status") == "completed":
        return {
            "status": "completed",
            "analysis": {
                "summary": analysis["summary"],
                "strategy_used": analysis["strategy_used"],
                "timestamp": analysis["timestamp"]
            }
        }
    
    return {"status": "processing", "message": "Analysis in progress..."}

@app.get("/api/debug/sessions")
async def debug_sessions():
    """Debug endpoint to check sessions"""
    return {
        "session_count": len(sessions),
        "sessions": {k: {
            "created_at": str(v.get("created_at")),
            "analysis_status": v.get("analysis", {}).get("status", "none"),
            "file_id": v.get("analysis", {}).get("file_id", "none")
        } for k, v in sessions.items()}
    }

@app.post("/api/chat", response_model=ChatResponse)
async def chat_with_report(
    request: ChatRequest,
    session_id: str = Depends(get_session_id)
):
    """
    Chat with the AI about the uploaded inspection report
    """
    try:
        # Use session_id from request body if provided, otherwise from auth header
        actual_session_id = request.session_id or session_id
        
        logger.info(f"Chat request - Request session: {request.session_id}, Auth session: {session_id}, Using: {actual_session_id}")
        
        session = sessions.get(actual_session_id)
        if not session:
            logger.error(f"Session not found: {actual_session_id}, Available sessions: {list(sessions.keys())}")
            raise HTTPException(status_code=404, detail="Session not found")
        
        analysis = session.get("analysis")
        logger.info(f"Session analysis status: {analysis.get('status') if analysis else 'No analysis'}")
        
        if not analysis or analysis.get("status") != "completed":
            raise HTTPException(
                status_code=400, 
                detail="No completed report analysis found. Please upload a report first."
            )
        
        # Add report context to the question
        enhanced_question = f"""
        Based on the California property inspection report that was analyzed, please answer this question:
        
        {request.message}
        
        Context from the inspection report:
        {analysis['pdf_text'][:4000]}
        
        Previous analysis summary:
        {analysis['summary'][:2000]}
        """
        
        # Use enhanced agent for follow-up questions
        response = await analyzer.chat(enhanced_question)
        
        # Update conversation history
        session["conversation_history"].append({
            "timestamp": datetime.now().isoformat(),
            "question": request.message,
            "answer": response["answer"],
            "strategy": response["strategy"]
        })
        
        logger.info(f"💬 Chat response generated (Session: {session_id[:8]})")
        
        return ChatResponse(
            message=response["answer"],
            strategy=response["strategy"],
            timestamp=datetime.now().isoformat(),
            session_id=session_id
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Chat error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Chat failed: {str(e)}")

@app.get("/api/session-info", response_model=RateLimitInfo)
async def get_session_info(session_id: str = Depends(get_session_id)):
    """Get information about the current session and rate limits"""
    session = sessions.get(session_id)
    if not session:
        # Create new session
        sessions[session_id] = {
            "created_at": datetime.now(),
            "reports_uploaded": 0,
            "conversation_history": [],
            "current_report": None
        }
        session = sessions[session_id]
    
    remaining_reports = max(0, 3 - session["reports_uploaded"])
    
    return RateLimitInfo(
        session_id=session_id,
        reports_uploaded_today=session["reports_uploaded"],
        remaining_reports=remaining_reports,
        conversation_count=len(session["conversation_history"]),
        has_active_report=bool(session.get("analysis", {}).get("status") == "completed")
    )

@app.delete("/api/session")
async def clear_session(session_id: str = Depends(get_session_id)):
    """Clear the current session"""
    if session_id in sessions:
        del sessions[session_id]
    return {"message": "Session cleared successfully"}

if __name__ == "__main__":
    uvicorn.run(
        "main:app", 
        host="127.0.0.1", 
        port=8000, 
        reload=True,
        log_level="info"
    )