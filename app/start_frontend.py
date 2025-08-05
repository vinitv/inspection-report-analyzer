#!/usr/bin/env python3
"""
Start the Streamlit frontend
"""

import os
import sys
import subprocess
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Setup environment
from dotenv import load_dotenv
load_dotenv()

if __name__ == "__main__":
    print("🎨 Starting California Property Inspection Analyzer Frontend...")
    print("🌐 Frontend will be available at: http://localhost:8501")
    print("⚠️  Make sure the backend is running on: http://127.0.0.1:8000")
    print("-" * 80)
    
    # Start Streamlit
    frontend_path = project_root / "app" / "frontend" / "streamlit_app.py"
    
    subprocess.run([
        "uv", "run", "streamlit", "run",
        str(frontend_path),
        "--server.port", "8501",
        "--server.address", "localhost",
        "--browser.gatherUsageStats", "false"
    ])