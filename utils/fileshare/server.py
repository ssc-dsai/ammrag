import http.server
import socketserver
import os

PORT = 8642
PUB_DIR = "/app/pub"

os.chdir(PUB_DIR)

Handler = http.server.SimpleHTTPRequestHandler

with socketserver.TCPServer(("", PORT), Handler) as httpd:
    print(f"Serving {PUB_DIR} on port {PORT}")
    httpd.serve_forever()
