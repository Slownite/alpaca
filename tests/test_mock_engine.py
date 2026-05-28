import time
import requests
import pytest
import requests
from alpaca.engine.mock import MockEngine

def test_mock_engine_lifecycle_smoke():
    """Smoke test exercising start -> health -> request -> stop with no GPU."""
    
    # 1. Initialize the engine on an open local port
    engine = MockEngine()
    test_model = "meta-llama/Llama-3-8B-Instruct"
    test_port = 8095
    
    try:
        # 2. Start the engine (spins up the background thread)
        engine.start(model=test_model, port=test_port)
        
        # Give the server thread a tiny split second to bind to the port
        time.sleep(0.1)
        
        # 3. Assert Health
        assert engine.health() is True
        assert engine.endpoint() == f"http://localhost:{test_port}"
        
        # 4. Exercise Request (Non-Streaming POST)
        endpoint_url = f"{engine.endpoint()}/v1/chat/completions"
        payload = {
            "model": test_model,
            "messages": [{"role": "user", "content": "Ping"}],
            "stream": False
        }
        
        response = requests.post(endpoint_url, json=payload, timeout=5)
        assert response.status_code == 200
        assert response.headers["Content-Type"] == "application/json"
        
        data = response.json()
        assert data["model"] == test_model
        assert "choices" in data
        assert len(data["choices"]) > 0

        # 5. Exercise Request (Streaming SSE with Measurable Delay)
        payload["stream"] = True
        
        start_time = time.time()
        # stream=True forces requests to stream the socket chunks line by line
        stream_response = requests.post(endpoint_url, json=payload, stream=True, timeout=5)
        assert stream_response.status_code == 200
        assert stream_response.headers["Content-Type"] == "text/event-stream"
        
        lines = []
        for line in stream_response.iter_lines():
            if line:
                lines.append(line.decode("utf-8"))
        
        elapsed_time = time.time() - start_time
        
        # Verify that we actually received multiple chunks ending with [DONE]
        assert len(lines) > 2
        assert lines[-1] == "data: [DONE]"
        
        # Verify measurable per-token delay occurred (e.g., should take at least 0.3-0.5s for tokens)
        assert elapsed_time > 0.2

    finally:
        # 6. Stop Engine (Clean up background threads and sockets even if assertions fail)
        engine.stop()
        
        # Wait a moment for teardown to settle and re-verify health
        time.sleep(0.1)
        assert engine.health() is False