import http.server
import socketserver

PORT = 8080
DIRECTORY = "simple_client"

class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIRECTORY, **kwargs)

print(f"Starting client server on port {PORT}...")
print(f"Open http://localhost:{PORT} in your browser")

with socketserver.TCPServer(("", PORT), Handler) as httpd:
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down client server...")
        httpd.server_close()
