from __future__ import annotations

from alpaca.engine.base import Engine
import http.server
import socketserver
import json
import threading

class OpenAIStubHandler(http.server.BaseHTTPRequestHandler):
    def do_POST(self):
        # Handle chat completions or normal completions endpoints
        if self.path in ["/v1/chat/completions", "/v1/completions"]:
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            
            # Simple mockup response structure matching OpenAI API expectations
            mock_response = {
                "id": "cmpl-mock",
                "object": "text_completion",
                "created": 1700000000,
                "model": "mock-stub-model",
                "choices": [
                    {
                        "text": "Stub response: vLLM is mocked out on CPU!",
                        "index": 0,
                        "logprobs": None,
                        "finish_reason": "stop"
                    }
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
            }
            
            self.wfile.write(json.dumps(mock_response).encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()
        
    def do_GET(self):
        if self.path == "/":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            mock_response = {"status": "mock engine running"}
            self.wfile.write(json.dumps(mock_response).encode("utf-8"))
        if self.path == "/v1/models":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            mock_response = {"data": [{"id": "mock-stub-model", "object": "model"}]}
            self.wfile.write(json.dumps(mock_response).encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()

    # Silences annoying terminal spam for every local health check/request
    def log_message(self, format, *args):
        return

class MockEngine(Engine):
    """OpenAI-compatible stub with no model and no GPU.

    Lets the CLI and Ray orchestration run locally and in CI; only real
    benchmark runs need hardware.
    """

    def start(self, model: str, port: int, **opts) -> None:
        self.model = model
        self.port = port
        self.opts = opts
        self.handler = OpenAIStubHandler
        socketserver.TCPServer.allow_reuse_address = True
        self.server = socketserver.TCPServer(("localhost", self.port), self.handler)
        self.server_thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.server_thread.start()

    def stop(self) -> None:
        if self.server:
            self.server.shutdown()      # Stops the serve_forever loop
            self.server.server_close()  # Closes the socket clean
            self.server_thread.join()   # Cleans up the background thre

    def health(self) -> bool:
        return self.server is not None and self.server_thread.is_alive()

    def endpoint(self) -> str:
        return f"http://localhost:{self.port}/v1"
