#!/usr/bin/env python
"""
Simple launcher for HTML frontend
"""
import http.server
import socketserver
import os
import webbrowser
import threading
import time
from pathlib import Path

def start_frontend():
    """Start the HTML frontend"""
    # Change to frontend directory
    frontend_dir = Path(__file__).parent / "app" / "frontend"
    os.chdir(frontend_dir)
    
    port = 3000
    
    print(f"🌐 Starting HTML frontend...")
    print(f"📁 Directory: {frontend_dir}")
    print(f"📄 Files: {list(frontend_dir.glob('*'))}")
    
    class CORSHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
        def end_headers(self):
            self.send_header('Access-Control-Allow-Origin', '*')
            self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
            self.send_header('Access-Control-Allow-Headers', 'Content-Type')
            super().end_headers()
            
        def do_OPTIONS(self):
            self.send_response(200)
            self.end_headers()
    
    try:
        with socketserver.TCPServer(("", port), CORSHTTPRequestHandler) as httpd:
            print(f"✅ HTML Frontend running at: http://localhost:{port}")
            print(f"🚀 Open this URL in your browser!")
            print(f"⚠️  Make sure backend is running at: http://localhost:8000")
            print(f"📚 Backend API docs: http://localhost:8000/api/docs")
            print("\nPress Ctrl+C to stop")
            
            # Try to open browser
            def open_browser():
                time.sleep(2)
                try:
                    webbrowser.open(f'http://localhost:{port}')
                    print(f"🌐 Browser opened automatically")
                except:
                    print(f"🌐 Please manually open: http://localhost:{port}")
            
            threading.Thread(target=open_browser, daemon=True).start()
            httpd.serve_forever()
            
    except KeyboardInterrupt:
        print("\n🛑 Frontend stopped")
    except OSError as e:
        if "Address already in use" in str(e):
            print(f"❌ Port {port} is already in use")
            print(f"🔧 Try: lsof -ti:{port} | xargs kill")
        else:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    start_frontend()