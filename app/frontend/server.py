#!/usr/bin/env python3
"""
Simple HTTP server to serve the HTML frontend
"""

import http.server
import socketserver
import os
import sys
from pathlib import Path

def start_frontend_server(port=3000):
    """Start the frontend server"""
    
    # Change to the frontend directory
    frontend_dir = Path(__file__).parent
    os.chdir(frontend_dir)
    
    print(f"🌐 Starting frontend server on http://localhost:{port}")
    print(f"📁 Serving files from: {frontend_dir}")
    print(f"🔗 Open http://localhost:{port} in your browser")
    print("Press Ctrl+C to stop the server")
    
    # Custom handler to set CORS headers
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
            httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n🛑 Server stopped")
    except OSError as e:
        if "Address already in use" in str(e):
            print(f"❌ Port {port} is already in use. Try a different port:")
            print(f"   python server.py {port + 1}")
        else:
            print(f"❌ Error starting server: {e}")

if __name__ == "__main__":
    port = 3000
    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[1])
        except ValueError:
            print("Usage: python server.py [port]")
            sys.exit(1)
    
    start_frontend_server(port)