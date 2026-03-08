"""
Tiny launcher service — uses almost zero resources.
Listens on port 5056. When it gets a /start request, it launches the heavy server.
Auto-starts with Windows via Task Scheduler.
"""
import subprocess
import sys
import os
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
import json
import threading

PORT = 5056
SERVER_PORT = 5055
SERVER_DIR = os.path.dirname(os.path.abspath(__file__))
server_process = None
server_lock = threading.Lock()


def is_server_running():
    """Check if the main server is responding."""
    import urllib.request
    try:
        r = urllib.request.urlopen(f'http://127.0.0.1:{SERVER_PORT}/health', timeout=2)
        return r.status == 200
    except Exception:
        return False


def start_server():
    """Launch the main server in a visible window."""
    global server_process
    with server_lock:
        if is_server_running():
            return True
        try:
            # Launch in a new visible console window
            # Use python.exe explicitly (not pythonw.exe) so server has full console I/O
            python_exe = sys.executable.replace('pythonw', 'python') if 'pythonw' in sys.executable else sys.executable
            server_process = subprocess.Popen(
                [python_exe, os.path.join(SERVER_DIR, 'server.py')],
                cwd=SERVER_DIR,
                creationflags=subprocess.CREATE_NEW_CONSOLE if os.name == 'nt' else 0,
            )
            # Wait for it to start
            for _ in range(60):
                time.sleep(1)
                if is_server_running():
                    return True
            return False
        except Exception as e:
            print(f'[LAUNCHER] Failed to start server: {e}')
            return False


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/start':
            ok = start_server()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps({'ok': ok, 'server': f'http://127.0.0.1:{SERVER_PORT}'}).encode())
        elif self.path == '/status':
            running = is_server_running()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps({'running': running}).encode())
        else:
            self.send_response(404)
            self.end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET')
        self.end_headers()

    def log_message(self, format, *args):
        pass  # Silent


if __name__ == '__main__':
    print(f'[LAUNCHER] Manga server launcher on port {PORT}')
    print(f'[LAUNCHER] GET /start  → starts the heavy server')
    print(f'[LAUNCHER] GET /status → checks if server is running')
    httpd = HTTPServer(('0.0.0.0', PORT), Handler)
    httpd.serve_forever()
