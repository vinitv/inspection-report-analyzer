"""
Utility functions for the Inspection Report Analyzer
"""

import os
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

from langchain_community.document_loaders import PyMuPDFLoader

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
        ]
    )

def process_pdf(file_path: Path) -> str:
    """
    Extract text from PDF file using PyMuPDFLoader
    
    Args:
        file_path: Path to the PDF file
        
    Returns:
        Extracted text content
    """
    try:
        loader = PyMuPDFLoader(str(file_path))
        documents = loader.load()
        
        # Combine all pages into single text
        text_content = ""
        for doc in documents:
            text_content += doc.page_content + "\n\n"
        
        # Clean up the text
        text_content = text_content.strip()
        
        # Log extraction info
        logging.info(f"📄 Extracted {len(text_content)} characters from {len(documents)} pages")
        
        return text_content
        
    except Exception as e:
        logging.error(f"❌ PDF processing error: {str(e)}")
        raise Exception(f"Failed to process PDF: {str(e)}")

class RateLimiter:
    """
    Simple rate limiter for PDF uploads
    """
    
    def __init__(self, max_reports_per_day: int = 3):
        self.max_reports_per_day = max_reports_per_day
        self.user_uploads: Dict[str, List[datetime]] = {}
    
    def can_upload(self, session_id: str, current_count: int) -> bool:
        """
        Check if user can upload another report
        
        Args:
            session_id: User session identifier
            current_count: Current number of reports uploaded today
            
        Returns:
            True if user can upload, False otherwise
        """
        return current_count < self.max_reports_per_day
    
    def record_upload(self, session_id: str):
        """Record a successful upload for rate limiting"""
        now = datetime.now()
        
        if session_id not in self.user_uploads:
            self.user_uploads[session_id] = []
        
        self.user_uploads[session_id].append(now)
        
        # Clean up old entries (older than 24 hours)
        cutoff = now - timedelta(days=1)
        self.user_uploads[session_id] = [
            upload_time for upload_time in self.user_uploads[session_id]
            if upload_time > cutoff
        ]
    
    def get_remaining_uploads(self, session_id: str) -> int:
        """Get remaining uploads for the day"""
        if session_id not in self.user_uploads:
            return self.max_reports_per_day
        
        # Clean up old entries
        now = datetime.now()
        cutoff = now - timedelta(days=1)
        recent_uploads = [
            upload_time for upload_time in self.user_uploads[session_id]
            if upload_time > cutoff
        ]
        
        return max(0, self.max_reports_per_day - len(recent_uploads))

def validate_pdf_file(file_content: bytes, filename: str) -> bool:
    """
    Validate PDF file
    
    Args:
        file_content: File content bytes
        filename: Original filename
        
    Returns:
        True if valid PDF, False otherwise
    """
    # Check file extension
    if not filename.lower().endswith('.pdf'):
        return False
    
    # Check PDF magic bytes
    if not file_content.startswith(b'%PDF-'):
        return False
    
    # Check file size (max 10MB)
    if len(file_content) > 10 * 1024 * 1024:
        return False
    
    return True

def clean_text_for_analysis(text: str) -> str:
    """
    Clean extracted text for better analysis
    
    Args:
        text: Raw extracted text
        
    Returns:
        Cleaned text
    """
    import re
    
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove page numbers and headers/footers patterns
    text = re.sub(r'Page \d+ of \d+', '', text, flags=re.IGNORECASE)
    text = re.sub(r'^\d+\s*$', '', text, flags=re.MULTILINE)
    
    # Clean up common OCR artifacts
    text = re.sub(r'[^\w\s\.\,\!\?\-\(\)\[\]\:;]', '', text)
    
    return text.strip()

def extract_priority_items(analysis_text: str) -> List[Dict[str, str]]:
    """
    Extract priority items from analysis text
    
    Args:
        analysis_text: Full analysis text from agent
        
    Returns:
        List of priority items with details
    """
    import re
    
    priority_items = []
    
    # Look for numbered priority items or bullet points
    patterns = [
        r'(\d+)\.\s*([^\n]+(?:\n(?!\d+\.)[^\n]+)*)',  # Numbered lists
        r'[•\-\*]\s*([^\n]+(?:\n(?![•\-\*])[^\n]+)*)',  # Bullet points
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, analysis_text, re.MULTILINE)
        for match in matches:
            if isinstance(match, tuple):
                item_text = match[1] if len(match) > 1 else match[0]
            else:
                item_text = match
            
            # Extract cost if mentioned
            cost_match = re.search(r'\$[\d,]+(?:\s*-\s*\$[\d,]+)?', item_text)
            cost = cost_match.group(0) if cost_match else "Cost estimate needed"
            
            # Determine urgency
            urgency = "Medium"
            if any(word in item_text.lower() for word in ['urgent', 'immediate', 'safety', 'danger']):
                urgency = "High"
            elif any(word in item_text.lower() for word in ['cosmetic', 'minor', 'aesthetic']):
                urgency = "Low"
            
            priority_items.append({
                "item": item_text.strip(),
                "cost": cost,
                "urgency": urgency
            })
    
    return priority_items[:5]  # Return top 5 items

def format_cost_estimate(cost_string: str) -> str:
    """
    Format cost estimates consistently
    
    Args:
        cost_string: Raw cost string
        
    Returns:
        Formatted cost string
    """
    import re
    
    # Extract numbers from cost string
    numbers = re.findall(r'\d+(?:,\d{3})*', cost_string)
    
    if len(numbers) >= 2:
        return f"${numbers[0]} - ${numbers[1]}"
    elif len(numbers) == 1:
        return f"~${numbers[0]}"
    else:
        return "Cost estimate needed"

def create_inspection_summary(analysis: str, priority_items: List[Dict]) -> Dict[str, any]:
    """
    Create a structured summary of the inspection analysis
    
    Args:
        analysis: Full analysis text
        priority_items: List of priority items
        
    Returns:
        Structured summary dictionary
    """
    import re
    
    # Extract overall condition
    condition_keywords = {
        "Excellent": ["excellent", "perfect", "no issues", "great condition"],
        "Good": ["good", "minor issues", "well maintained"],
        "Fair": ["fair", "moderate", "some concerns", "attention needed"],
        "Poor": ["poor", "major issues", "significant problems", "immediate attention"]
    }
    
    overall_condition = "Fair"  # Default
    analysis_lower = analysis.lower()
    
    for condition, keywords in condition_keywords.items():
        if any(keyword in analysis_lower for keyword in keywords):
            overall_condition = condition
            break
    
    # Count different types of issues
    safety_count = len([item for item in priority_items if item.get('urgency') == 'High'])
    cosmetic_count = len([item for item in priority_items if item.get('urgency') == 'Low'])
    
    # Extract total cost estimate
    total_cost = "Contact contractors for detailed estimates"
    cost_numbers = re.findall(r'\$[\d,]+', analysis)
    if cost_numbers:
        try:
            numbers = [int(re.sub(r'[^\d]', '', cost)) for cost in cost_numbers]
            if numbers:
                total_range = f"${min(numbers):,} - ${max(numbers):,}"
                total_cost = total_range
        except:
            pass
    
    return {
        "overall_condition": overall_condition,
        "total_priority_items": len(priority_items),
        "safety_issues": safety_count,
        "cosmetic_issues": cosmetic_count,
        "estimated_total_cost": total_cost,
        "analysis_timestamp": datetime.now().isoformat()
    }