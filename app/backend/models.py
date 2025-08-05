"""
Pydantic models for the Inspection Report Analyzer API
"""

from typing import List, Optional, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field

class ChatRequest(BaseModel):
    """Request model for chat interactions"""
    message: str = Field(..., min_length=1, max_length=2000, description="User's question about the inspection report")
    session_id: Optional[str] = Field(None, description="Session identifier")

class ChatResponse(BaseModel):
    """Response model for chat interactions"""
    message: str = Field(..., description="AI assistant's response")
    strategy: str = Field(..., description="Strategy used by the agent (TOOL_SEARCH, RAG_SIMPLE, RAG_COMPLEX)")
    timestamp: str = Field(..., description="Response timestamp")
    session_id: str = Field(..., description="Session identifier")

class UploadResponse(BaseModel):
    """Response model for PDF upload"""
    file_id: str = Field(..., description="Unique identifier for the uploaded file")
    filename: str = Field(..., description="Original filename")
    status: str = Field(..., description="Processing status (processing, completed, error)")
    message: str = Field(..., description="Status message")
    session_id: str = Field(..., description="Session identifier")

class PriorityItem(BaseModel):
    """Model for priority repair items"""
    item: str = Field(..., description="Description of the repair item")
    category: str = Field(..., description="Category (Safety, Structural, Electrical, etc.)")
    urgency: str = Field(..., description="Urgency level (High, Medium, Low)")
    estimated_cost: str = Field(..., description="Estimated cost range")
    timeline: str = Field(..., description="Recommended timeline for repair")
    description: str = Field(..., description="Detailed description and recommendations")

class ReportAnalysis(BaseModel):
    """Model for complete report analysis"""
    summary: str = Field(..., description="Executive summary of the inspection")
    priority_items: List[PriorityItem] = Field(..., description="Top priority items requiring attention")
    total_estimated_cost: str = Field(..., description="Total estimated cost for all priority items")
    overall_property_condition: str = Field(..., description="Overall assessment of property condition")
    california_specific_notes: List[str] = Field(default=[], description="California-specific considerations")
    safety_concerns: List[str] = Field(default=[], description="Immediate safety concerns")
    cosmetic_items: List[str] = Field(default=[], description="Non-urgent cosmetic items")
    analysis_timestamp: str = Field(..., description="When the analysis was completed")
    file_id: str = Field(..., description="Associated file ID")

class ErrorResponse(BaseModel):
    """Error response model"""
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    timestamp: str = Field(..., description="Error timestamp")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")

class RateLimitInfo(BaseModel):
    """Rate limiting information"""
    session_id: str = Field(..., description="Session identifier")
    reports_uploaded_today: int = Field(..., description="Number of reports uploaded today")
    remaining_reports: int = Field(..., description="Remaining reports allowed today")
    conversation_count: int = Field(..., description="Number of chat messages in this session")
    has_active_report: bool = Field(..., description="Whether there's an active report for analysis")

class SessionInfo(BaseModel):
    """Session information model"""
    session_id: str = Field(..., description="Session identifier")
    created_at: str = Field(..., description="Session creation timestamp")
    last_activity: str = Field(..., description="Last activity timestamp")
    current_report: Optional[str] = Field(None, description="Current active report ID")
    conversation_history: List[Dict[str, Any]] = Field(default=[], description="Chat history")

class HealthStatus(BaseModel):
    """Health check response"""
    status: str = Field(..., description="Service status")
    timestamp: str = Field(..., description="Health check timestamp")
    version: str = Field(..., description="API version")
    analyzer_status: str = Field(..., description="Analyzer component status")
    database_status: Optional[str] = Field(None, description="Database status if applicable")

class AgentResponse(BaseModel):
    """Internal model for agent responses"""
    answer: str = Field(..., description="Agent's answer")
    strategy: str = Field(..., description="Strategy used by the agent")
    reasoning: str = Field(..., description="Agent's reasoning")
    context: Optional[str] = Field(None, description="Retrieved context")
    tool_calls: List[Dict[str, Any]] = Field(default=[], description="Tools called during processing")