from __future__ import annotations

from alpaca.engine.base import Engine
import http.server
import socketserver
import json
import threading
import time
class OpenAIStubHandler(http.server.BaseHTTPRequestHandler):
    # HTTP/1.1 so streamed responses without Content-Length get framed as
    # chunked transfer encoding; under the default HTTP/1.0 the client cannot
    # detect end-of-body and blocks until its read timeout.
    protocol_version = "HTTP/1.1"

    def send_json_response(self, data: dict, status: int = 200):
        response_bytes = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(response_bytes)))
        self.end_headers()
        self.wfile.write(response_bytes)

    def do_POST(self):
        current_model = getattr(self.server, "engine_instance").model

        if self.path in ["/v1/chat/completions", "/v1/completions"]:
            # 1. Parse the incoming request body to check if stream is requested
            content_length = int(self.headers.get('Content-Length', 0))
            request_body = self.rfile.read(content_length).decode('utf-8') if content_length else "{}"
            try:
                request_data = json.loads(request_body)
            except json.JSONDecodeError:
                request_data = {}

            is_streaming = request_data.get("stream", False)

            if is_streaming:
                # --- START STREAMING (SSE) LOGIC ---
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                # No Content-Length on a streamed body, so signal end-of-response
                # by closing the socket; otherwise the HTTP/1.1 client blocks
                # waiting for more data until its read timeout fires.
                self.send_header("Connection", "close")
                self.close_connection = True
                self.end_headers()

                full_text = "Stub response: vLLM is mocked out on CPU with live token streaming!"
                tokens = full_text.split(" ")  # Simulate tokenization by splitting on spaces

                for index, token in enumerate(tokens):
                    # Add space back to words except the first one to mimic real generation
                    token_to_send = f" {token}" if index > 0 else token
                    
                    # Structure matching OpenAI's streaming chunk payload
                    chunk = {
                        "id": "chatcmpl-mock-stream",
                        "object": "chat.completion.chunk",
                        "created": 1700000000,
                        "model": current_model,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"content": token_to_send},
                                "finish_reason": None
                            }
                        ]
                    }
                    
                    # Format as SSE event string: data: {json}\n\n
                    self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode("utf-8"))
                    self.wfile.flush() # Force transmission down the socket immediately
                    
                    # Measurable per-token delay (e.g., 50ms per token)
                    time.sleep(0.05)

                # Send final chunk indicating generation stopped
                final_chunk = {
                    "id": "chatcmpl-mock-stream",
                    "object": "chat.completion.chunk",
                    "created": 1700000000,
                    "model": current_model,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]
                }
                self.wfile.write(f"data: {json.dumps(final_chunk)}\n\n".encode("utf-8"))
                
                # Signal completion with the exact OpenAI [DONE] delimiter
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
                
            else:
                # --- STANDARD NON-STREAMING LOGIC ---
                mock_response = {
                    "id": "cmpl-mock",
                    "object": "text_completion",
                    "created": 1700000000,
                    "model": current_model,
                    "choices": [
                        {
                            "text": "Stub response: vLLM is mocked out on CPU!",
                            "index": 0,
                            "finish_reason": "stop"
                        }
                    ],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
                }
                self.send_json_response(mock_response)
        else:
            self.send_json_response({"error": "Not Found"}, status=404)
        
    def do_GET(self):
        current_model = getattr(self.server, "engine_instance").model
        if self.path == "/":
            self.send_json_response({"status": "mock engine running", "model": current_model})
        elif self.path == "/v1/models":
            self.send_json_response({"data": [{"id": current_model, "object": "model"}]})
        else:
            self.send_json_response({"error": "Not Found"}, status=404)

    # Silences annoying terminal spam for every local health check/request
    def log_message(self, format, *args):
        return

class MockEngine(Engine):
    """OpenAI-compatible stub with no model and no GPU."""

    def start(self, model: str, port: int, **opts) -> None:
        self.model = model
        self.port = port
        self.opts = opts
        self.handler = OpenAIStubHandler

        self.server = socketserver.ThreadingTCPServer(
            ("localhost", self.port), self.handler, bind_and_activate=False
        )
        self.server.allow_reuse_address = True
        # Daemonize worker threads so server_close() doesn't block joining a
        # request thread parked in readinto() on an idle keep-alive connection.
        self.server.daemon_threads = True
        self.server.server_bind()
        self.server.server_activate()

        self.server.engine_instance = self

        self.server_thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.server_thread.start()

    def stop(self) -> None:
        if self.server:
            self.server.shutdown()
            self.server.server_close()
            self.server_thread.join()

    def health(self) -> bool:
        return self.server is not None and self.server_thread.is_alive()

    def endpoint(self) -> str:
        return f"http://localhost:{self.port}"