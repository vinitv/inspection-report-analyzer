#!/usr/bin/env python3
"""
Complete launcher for HTML-based Property Inspection Analyzer
Starts backend and HTML frontend
"""

import os
import sys
import time
import subprocess
import signal
import webbrowser
import threading
from pathlib import Path

def setup_environment():
    """Setup environment variables"""
    from dotenv import load_dotenv
    load_dotenv()
    
    required_keys = ["OPENAI_API_KEY"]
    missing_keys = []
    
    for key in required_keys:
        if not os.getenv(key):
            missing_keys.append(key)
    
    if missing_keys:
        print("❌ Missing required API keys:")
        for key in missing_keys:
            print(f"   - {key}")
        print("\n📝 Please copy env_template.txt to .env and add your API keys")
        return False
    
    print("✅ Environment setup complete")
    return True

def main():
    """Main launcher for complete HTML app"""
    print("🏠 Property Inspection Analyzer - HTML Edition")
    print("=" * 60)
    
    # Setup environment
    if not setup_environment():
        sys.exit(1)
    
    # Start backend server
    print("\n🚀 Starting backend server...")
    backend_process = subprocess.Popen([
        "uv", "run", "python", "-c",
        "import sys; sys.path.insert(0, '.'); "
        "import uvicorn; "
        "from app.backend.main import app; "
        "print('🔧 Backend API running on: http://127.0.0.1:8000'); "
        "print('📊 API Documentation: http://127.0.0.1:8000/api/docs'); "
        "uvicorn.run(app, host='127.0.0.1', port=8000, log_level='info')"
    ])
    
    # Wait for backend to start
    print("⏳ Waiting for backend to initialize...")
    time.sleep(8)
    
    # Start HTML frontend
    print("\n🎨 Starting HTML frontend...")
    frontend_process = subprocess.Popen([
        "uv", "run", "python", "launch_html.py"
    ])
    
    # Wait a moment for frontend to start
    time.sleep(3)
    
    print("\n" + "=" * 60)
    print("🎉 Complete Application Started Successfully!")
    print("\n🌟 **NEW HTML INTERFACE**")
    print("🌐 Frontend: http://localhost:3000")
    print("🔧 Backend:  http://127.0.0.1:8000")
    print("📚 API Docs: http://127.0.0.1:8000/api/docs")
    print("\n✨ **Features:**")
    print("   • Modern drag & drop file upload")
    print("   • Real-time chat interface")
    print("   • Cost estimation tools")
    print("   • Professional UI design")
    print("   • Mobile responsive")
    print("\n💡 **Usage:**")
    print("   1. Open http://localhost:3000 in your browser")
    print("   2. Drag & drop a PDF inspection report")
    print("   3. Click sample questions or type your own")
    print("   4. Get real-time analysis and cost estimates")
    print("\n⚠️  Press Ctrl+C to stop both servers")
    print("=" * 60)
    
    # Open browser automatically
    def open_browser():
        time.sleep(2)
        try:
            webbrowser.open('http://localhost:3000')
            print("\n🌐 Browser opened automatically to HTML frontend!")
        except:
            print("\n🌐 Please manually open: http://localhost:3000")
    
    threading.Thread(target=open_browser, daemon=True).start()
    
    try:
        # Wait for processes
        while True:
            time.sleep(1)
            
            # Check if processes are still running
            if backend_process.poll() is not None:
                print("❌ Backend process stopped")
                break
            if frontend_process.poll() is not None:
                print("❌ Frontend process stopped")
                break
                
    except KeyboardInterrupt:
        print("\n🛑 Shutting down...")
        
        # Gracefully terminate processes
        print("   Stopping frontend...")
        frontend_process.terminate()
        
        print("   Stopping backend...")
        backend_process.terminate()
        
        # Wait for processes to stop
        try:
            frontend_process.wait(timeout=5)
            backend_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            print("   Force killing processes...")
            frontend_process.kill()
            backend_process.kill()
        
        print("✅ Application stopped successfully")
    
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        
        # Cleanup processes
        try:
            frontend_process.terminate()
            backend_process.terminate()
        except:
            pass

if __name__ == "__main__":
    main()