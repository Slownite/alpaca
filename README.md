# 🦙 Alpaca — Ollama-style vLLM Wrapper

A streamlined CLI that brings Ollama's simplicity to [vLLM](https://github.com/vllm-project/vllm) using Docker containers, with optional distributed serving via [Ray](https://ray.io/). Pull, serve, and manage large language models with ease.

> Requires Docker and Python 3.9+ ([download](https://www.python.org/downloads/)). Licensed under [MIT](https://opensource.org/licenses/MIT).

---

## ✨ Features

### 🚀 Simple model management

* **Pull & cache** models from Hugging Face
* **Serve locally** with automatic CPU/GPU detection via Docker
* **Process control**: list, stop, and manage Docker containers
* **Aliases** for friendly model names

### 🐳 Docker-based serving

* **Pre-built containers** for immediate deployment
* **Automatic hardware detection** (CPU/GPU)
* **Resource isolation** and management
* **Easy port management** and networking

### 🌐 Distributed inference

* **Ray integration** for multi-node scaling
* **Cluster orchestration** (head/worker helpers)
* **Seamless multi-GPU / multi-node** deployments

---

## 🚀 Quick Start

> In the examples below we use the `alpaca` CLI. If you're running directly from the repo without installing, replace `alpaca` with `python alpaca.py`.

### Prerequisites

Make sure you have Docker installed and running:
```bash
docker version  # Should show client and server info
```

### Installation (from source)

```bash
git clone https://github.com/Slownite/alpaca.git
cd alpaca
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
pip install -e .
```

### Serve your first model

```bash
# Pull a model from Hugging Face and alias it
alpaca pull microsoft/DialoGPT-medium --alias chatbot

# Start serving (automatically detects CPU/GPU)
alpaca serve chatbot

# Test a prompt
alpaca run chatbot -p "Hello, how are you today?"
```

---

## 📋 Command Reference

### 🦙 Model Management

| Command | Description               | Example                                 |
| ------- | ------------------------- | --------------------------------------- |
| `pull`  | Download model from HF    | `alpaca pull meta-llama/Llama-2-7b-hf`  |
| `ls`    | List cached models        | `alpaca ls --size`                      |
| `serve` | Start model server        | `alpaca serve llama2 --port 8000`       |
| `ps`    | List running containers   | `alpaca ps`                             |
| `stop`  | Stop a server container   | `alpaca stop llama2`                    |
| `rm`    | Remove container/alias    | `alpaca rm llama2 --purge`              |
| `run`   | Send test request         | `alpaca run llama2 -p "Tell me a joke"` |
| `logs`  | Show container logs       | `alpaca logs llama2 -f`                 |
| `config`| Show/set configuration    | `alpaca config --show`                  |

**Serve options**

* `--gpu` / `--cpu` force mode
* `--port PORT` specify port (default: auto)
* `--dtype {auto,float32,bf16,fp16}` precision
* `--max-seqs N` max concurrent sequences

### 🌐 Distributed (Ray)

| Command        | Description                    | Example                                              |
| -------------- | ------------------------------ | ---------------------------------------------------- |
| `ray-head`     | Start Ray head node            | `alpaca ray-head --dashboard-port 8265`             |
| `ray-worker`   | Start Ray worker               | `alpaca ray-worker --address head:6379 --gpu`       |
| `serve-ray`    | Serve with Ray backend         | `alpaca serve-ray llama2 --address ray://head:10001` |
| `ray-down`     | Stop Ray head/workers          | `alpaca ray-down`                                    |
| `cluster-up`   | Start local Ray cluster        | `alpaca cluster-up --gpu-workers 2`                 |
| `cluster-down` | Stop local Ray cluster         | `alpaca cluster-down`                                |

---

## 📖 Usage Examples

### Local development

```bash
alpaca pull facebook/opt-125m --alias tiny
alpaca serve tiny --cpu
alpaca run tiny -p "The future of AI is"
```

### Production serving

```bash
alpaca pull meta-llama/Llama-2-7b-chat-hf --alias llama2-chat
alpaca serve llama2-chat --gpu --dtype bf16 --max-seqs 64
```

### Multi-node deployment

```bash
# On head node
alpaca ray-head --dashboard-port 8265

# On worker nodes
alpaca ray-worker --address HEAD_IP:6379 --gpu

# Serve distributed model
alpaca serve-ray llama2-70b --address ray://HEAD_IP:10001
```

### Custom API requests

```bash
# Send custom JSON payload
echo '{"model":"llama2","messages":[{"role":"user","content":"Explain quantum computing"}],"max_tokens":100}' > request.json
alpaca run llama2 --json request.json

# Use a different API endpoint
alpaca run llama2 --path /v1/completions -p "Once upon a time"
```

---

## 🔧 Configuration

### Environment variables

```bash
export ALPACA_IMAGE="vllm/vllm-openai:latest"        # vLLM Docker image
export ALPACA_RAY_IMAGE="rayproject/ray:2.34.0"      # Ray Docker image
export ALPACA_CACHE_DIR="$HOME/.cache/huggingface"   # Model cache directory
export ALPACA_STATE_DIR="$HOME/.alpaca"              # State directory
export ALPACA_PORT_START=8000                        # Starting port for auto-allocation
export ALPACA_DTYPE_CPU="float32"                    # Default CPU dtype
export ALPACA_DTYPE_GPU="auto"                       # Default GPU dtype
export ALPACA_MAX_SEQS=32                            # Default max sequences
export HUGGING_FACE_HUB_TOKEN="your_token_here"      # For private models
```

### Config commands

```bash
# Show current configuration
alpaca config --show

# Set configuration values
alpaca config --set max_workers=4 --set timeout=300
```

---

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Alpaca CLI    │    │  Docker Engine  │    │  Hugging Face   │
│ • Model mgmt    │────│ • vLLM containers│────│ • Model storage │
│ • Process ctrl  │    │ • Resource mgmt │    │ • Tokenizers    │
│ • Ray orchestr. │    │ • Networking    │    │ • Configs       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
          │                        │
          ▼                        ▼
┌─────────────────┐    ┌─────────────────┐
│   Ray Cluster   │    │  Local Hardware │
│ • Multi-node    │    │ • CPU / GPU     │
│ • Load balance  │    │ • Memory        │
│ • Fault tol.    │    │ • Storage       │
└─────────────────┘    └─────────────────┘
```

---

## 🎯 Comparison

| Feature        | Alpaca         | Ollama       | vLLM CLI       | Transformers    |
| -------------- | -------------- | ------------ | -------------- | --------------- |
| Ease of Use    | ✅ Simple CLI   | ✅ Simple CLI | ❌ Complex      | ❌ Code required |
| HF Integration | ✅ Native       | ❌ Manual     | ✅ Native       | ✅ Native        |
| Docker Support | ✅ Built-in     | ❌ No         | ❌ Manual       | ❌ Manual        |
| Distributed    | ✅ Ray support  | ❌ No         | ✅ Manual setup | ❌ No            |
| Process Mgmt   | ✅ Built-in     | ✅ Built-in   | ❌ Manual       | ❌ Manual        |
| Performance    | ✅ vLLM backend | ❌ llama.cpp  | ✅ vLLM         | ❌ Basic         |
| Model Formats  | ✅ HF           | ✅ GGUF       | ✅ HF           | ✅ HF            |

---

## 🐛 Troubleshooting

### Docker issues

```bash
# Check Docker is running
docker version

# Check available images
docker images | grep vllm

# Clean up stopped containers
docker container prune
```

### Common issues

**Docker daemon not running**

```bash
# Start Docker service (varies by OS)
sudo systemctl start docker  # Linux
# Or start Docker Desktop app
```

**Port conflicts (address in use)**

```bash
alpaca ps
alpaca stop <model-name>
```

**GPU not detected**

```bash
# Check Docker has NVIDIA runtime
docker run --rm --gpus all nvidia/cuda:12.2.0-base nvidia-smi
```

**OOM during serving**

```bash
alpaca serve <model> --dtype fp16
alpaca serve <model> --max-seqs 16
```

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Commit: `git commit -m "Add amazing feature"`
4. Push: `git push origin feature/amazing-feature`
5. Open a Pull Request

### Development setup

```bash
git clone https://github.com/Slownite/alpaca.git
cd alpaca
python -m venv dev-env
source dev-env/bin/activate
pip install -e .
```

---

## 📄 License

This project is licensed under the MIT License. See the `LICENSE` file for details.

---

## 🙏 Acknowledgments

* [vLLM](https://github.com/vllm-project/vllm) — high-performance LLM inference
* [Ray](https://ray.io/) — distributed computing
* [Ollama](https://ollama.ai/) — inspiration for a simple CLI
* [Hugging Face](https://huggingface.co/) — models & tokenizers

---

## 📞 Support

* **Issues:** [https://github.com/Slownite/alpaca/issues](https://github.com/Slownite/alpaca/issues)
* **Discussions:** [https://github.com/Slownite/alpaca/discussions](https://github.com/Slownite/alpaca/discussions)
* **Documentation:** [https://github.com/Slownite/alpaca/wiki](https://github.com/Slownite/alpaca/wiki)

---

Alpaca combines the ease of Ollama with the power and flexibility of vLLM, making large-language-model deployment accessible to everyone. 🦙✨
