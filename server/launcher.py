"""
MangaVoice — Tiny launcher service (uses almost zero resources).
Listens on port 5056. Starts the heavy server on demand.
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
            python_exe = sys.executable.replace('pythonw', 'python') if 'pythonw' in sys.executable else sys.executable
            env = os.environ.copy()
            env['PYTHONIOENCODING'] = 'utf-8'
            # Ensure Homebrew paths are in PATH (launchd doesn't inherit shell PATH)
            brew_paths = '/opt/homebrew/bin:/usr/local/bin'
            if brew_paths not in env.get('PATH', ''):
                env['PATH'] = brew_paths + ':' + env.get('PATH', '/usr/bin:/bin')
            # Use server_lite.py (ONNX-only, lightweight) as the default
            server_script = os.path.join(SERVER_DIR, 'server_lite.py')
            if not os.path.isfile(server_script):
                server_script = os.path.join(SERVER_DIR, 'server.py')
            server_process = subprocess.Popen(
                [python_exe, server_script],
                cwd=SERVER_DIR,
                creationflags=subprocess.CREATE_NEW_CONSOLE if os.name == 'nt' else 0,
                env=env,
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


def stop_server():
    """Stop the main server."""
    global server_process
    with server_lock:
        try:
            import urllib.request
            urllib.request.urlopen(f'http://127.0.0.1:{SERVER_PORT}/shutdown', timeout=3)
        except Exception:
            pass
        if server_process and server_process.poll() is None:
            try:
                server_process.kill()
                server_process.wait(timeout=5)
            except Exception:
                pass
        server_process = None
        time.sleep(1)
        return not is_server_running()


class Handler(BaseHTTPRequestHandler):
    def _respond(self, data):
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', 'chrome-extension://*')
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

    def do_GET(self):
        if self.path == '/start':
            ok = start_server()
            self._respond({'ok': ok, 'server': f'http://127.0.0.1:{SERVER_PORT}'})
        elif self.path == '/stop':
            ok = stop_server()
            self._respond({'ok': ok})
        elif self.path == '/restart':
            stop_server()
            ok = start_server()
            self._respond({'ok': ok, 'server': f'http://127.0.0.1:{SERVER_PORT}'})
        elif self.path == '/status':
            self._respond({'running': is_server_running()})
        else:
            self.send_response(404)
            self.end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', 'chrome-extension://*')
        self.send_header('Access-Control-Allow-Methods', 'GET')
        self.end_headers()

    def log_message(self, format, *args):
        pass  # Silent


if __name__ == '__main__':
    print(f'[LAUNCHER] MangaVoice launcher on port {PORT}')
    print(f'[LAUNCHER] GET /start | /stop | /restart | /status')
    httpd = HTTPServer(('127.0.0.1', PORT), Handler)
    httpd.serve_forever()
