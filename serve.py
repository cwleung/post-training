#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
serve.py — Unified Development Server Launcher for DeepAgents & RL Platform
===========================================================================
Usage:
    python3 serve.py             # Launches on http://127.0.0.1:8000
    PORT=9000 python3 serve.py   # Custom port
"""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PORT = int(os.environ.get("PORT", "8000"))
HOST = os.environ.get("HOST", "127.0.0.1")

# Ensure sub-packages are in sys.path
for p in [str(ROOT), str(ROOT / "rl"), str(ROOT / "rlvr")]:
    if p not in sys.path:
        sys.path.insert(0, p)

def main():
    try:
        import uvicorn
        print(f"🚀 Starting Unified DeepAgents & RL FastAPI platform on http://{HOST}:{PORT}")
        uvicorn.run("web.app:app", host=HOST, port=PORT, reload=True)
    except ImportError:
        print("⚠️ uvicorn not found; falling back to basic http.server...")
        import http.server
        import socketserver
        os.chdir(str(ROOT / "web" / "dist"))
        with socketserver.TCPServer((HOST, PORT), http.server.SimpleHTTPRequestHandler) as httpd:
            print(f"Serving production frontend at http://{HOST}:{PORT}")
            httpd.serve_forever()

if __name__ == "__main__":
    main()