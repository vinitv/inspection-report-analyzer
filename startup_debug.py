#!/usr/bin/env python3
"""
Simple startup test script for debugging
"""

import os
import sys
import logging

# Setup basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_imports():
    """Test if all required modules can be imported"""
    try:
        logger.info("Testing imports...")
        
        # Test basic imports
        import fastapi
        import uvicorn
        import requests
        import pandas
        import numpy
        
        # Test app-specific imports
        from app.backend.models import ChatRequest, ChatResponse
        from app.backend.utils import setup_logging
        
        logger.info("✅ All imports successful")
        return True
        
    except ImportError as e:
        logger.error(f"❌ Import error: {e}")
        return False

def test_environment():
    """Test environment variables"""
    logger.info("Testing environment...")
    
    required_vars = ["OPENAI_API_KEY", "REPAIR_API_KEY", "LANGSMITH_API_KEY"]
    
    for var in required_vars:
        value = os.getenv(var)
        if value:
            logger.info(f"✅ {var}: {'*' * min(len(value), 8)}")
        else:
            logger.warning(f"⚠️ {var}: Not set")
    
    logger.info(f"PORT: {os.getenv('PORT', '8000')}")
    logger.info(f"HOST: {os.getenv('HOST', '0.0.0.0')}")
    logger.info(f"PYTHONPATH: {os.getenv('PYTHONPATH', 'Not set')}")

def test_app_startup():
    """Test if the app can start"""
    try:
        logger.info("Testing app startup...")
        
        # Import the app
        from app.backend.main import app
        
        logger.info("✅ App imported successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ App startup error: {e}")
        return False

if __name__ == "__main__":
    logger.info("🔍 Starting debug tests...")
    
    # Test imports
    if not test_imports():
        sys.exit(1)
    
    # Test environment
    test_environment()
    
    # Test app startup
    if not test_app_startup():
        sys.exit(1)
    
    logger.info("✅ All tests passed!")
