#!/usr/bin/env python3
"""
Start the FastAPI backend server
"""

import os
import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Setup environment
from dotenv import load_dotenv
load_dotenv()

# Import and start the backend
if __name__ == "__main__":
    import uvicorn
    from app.backend.main import app
    
    print("🚀 Starting California Property Inspection Analyzer Backend...")
    print("📊 API Documentation will be available at: http://127.0.0.1:8000/api/docs")
    print("🔧 Backend API running on: http://127.0.0.1:8000")
    print("⚠️  Make sure to start the frontend separately with: uv run streamlit run app/frontend/streamlit_app.py")
    print("-" * 80)
    
    uvicorn.run(
        "app.backend.main:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        log_level="info",
        reload_dirs=[str(project_root / "app")]
    )