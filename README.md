# 🦙 Alpaca — Ollama-style vLLM Wrapper

A streamlined CLI that brings Ollama's simplicity to [vLLM](https://github.com/vllm-project/vllm) with native Python processes and automated [Ray](https://ray.io/) cluster management. Pull, serve, and scale large language models across multiple nodes with ease.

> Requires CUDA GPU, Python 3.9+, vLLM, Ray (for distributed clusters), and dependencies. Licensed under [MIT](https://opensource.org/licenses/MIT).

---

## ✨ Features

### 🚀 Simple model management

* **Pull & cache** models from Hugging Face
* **Serve locally** with GPU acceleration via vLLM
* **Process control**: list, stop, and manage vLLM processes
* **Aliases** for friendly model names

### ⚡ Native vLLM serving

* **Direct vLLM integration** for maximum performance
* **GPU-only operation** for optimal inference speed
* **Native Python processes** with PID tracking
* **Automatic port allocation** and networking

### 🌐 Ray cluster management

* **Automated cluster lifecycle** - start, monitor, and stop Ray clusters
* **Multi-node scaling** with worker node management
* **Resource control** - specify CPUs/GPUs per worker
* **Auto-discovery** - seamless integration with local clusters
* **Distributed serving** for large models across nodes

---

## 🚀 Quick Start

> In the examples below we use the `alpaca` CLI. If you're running directly from the repo without installing, replace `alpaca` with `python alpaca.py`.

### Prerequisites

Make sure you have CUDA GPU support:
```bash
nvidia-smi  # Should show GPU info
python -c "import torch; print(torch.cuda.is_available())"  # Should print True
```

### Installation

#### Option 1: from source (recommended)

```bash
git clone https://github.com/Slownite/alpaca.git
cd alpaca
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
pip install vllm huggingface_hub httpx psutil ray
```

#### Option 2: Direct dependencies

```bash
pip install vllm huggingface_hub httpx psutil ray
# Then download alpaca.py and run directly
```

### Serve your first model

```bash
# Pull a model from Hugging Face and alias it
python alpaca.py pull microsoft/DialoGPT-medium --alias chatbot

# Start serving (GPU only)
python alpaca.py serve chatbot

# Test a prompt
python alpaca.py run chatbot -p "Hello, how are you today?"
```

---

## 📋 Command Reference

### 🦙 Model Management

| Command | Description               | Example                                            |
| ------- | ------------------------- | -------------------------------------------------- |
| `pull`  | Download model from HF    | `python alpaca.py pull meta-llama/Llama-2-7b-hf`  |
| `ls`    | List cached models        | `python alpaca.py ls --size`                      |
| `serve` | Start model server (GPU)  | `python alpaca.py serve llama2 --port 8000`       |
| `ps`    | List running processes    | `python alpaca.py ps`                             |
| `stop`  | Stop a server process     | `python alpaca.py stop llama2`                    |
| `rm`    | Remove process/alias      | `python alpaca.py rm llama2 --purge`              |
| `run`   | Send test request         | `python alpaca.py run llama2 -p "Tell me a joke"` |
| `logs`  | Show process info         | `python alpaca.py logs llama2`                    |
| `config`| Show/set configuration    | `python alpaca.py config --show`                  |

**Serve options**

* `--port PORT` specify port (default: auto)
* `--dtype {auto,float32,bf16,fp16}` precision
* `--max-seqs N` max concurrent sequences

### 🌐 Ray Cluster Management

| Command         | Description                               | Example                                                        |
| --------------- | ----------------------------------------- | -------------------------------------------------------------- |
| `ray-head`      | Start Ray head node                      | `python alpaca.py ray-head --dashboard-port 8265`             |
| `ray-worker`    | Add worker node to cluster               | `python alpaca.py ray-worker --address 127.0.0.1:6379 --gpus 1` |
| `ray-status`    | Show cluster status and resources        | `python alpaca.py ray-status`                                 |
| `ray-down`      | Stop Ray cluster and cleanup             | `python alpaca.py ray-down`                                   |
| `cluster-up`    | Start local cluster with workers        | `python alpaca.py cluster-up --gpu-workers 2`                 |
| `cluster-down`  | Stop local cluster                       | `python alpaca.py cluster-down`                               |

### 🚀 Distributed Inference

| Command        | Description                           | Example                                                           |
| -------------- | ------------------------------------- | ----------------------------------------------------------------- |
| `serve-ray`    | Serve with Ray backend               | `python alpaca.py serve-ray llama2` (auto-detects local cluster) |

**Ray cluster options**
* `--dashboard-port PORT` dashboard port (default: 8265)
* `--client-port PORT` Ray client port (default: 10001)  
* `--gcs-port PORT` Ray GCS port (default: 6379)
* `--cpus N` / `--gpus N` resources per worker

---

## 📖 Usage Examples

### Local development

```bash
python alpaca.py pull facebook/opt-125m --alias tiny
python alpaca.py serve tiny
python alpaca.py run tiny -p "The future of AI is"
```

### Production serving

```bash
python alpaca.py pull meta-llama/Llama-2-7b-chat-hf --alias llama2-chat
python alpaca.py serve llama2-chat --dtype bf16 --max-seqs 64
```

### Distributed Ray cluster

```bash
# Start local Ray cluster with 2 GPU workers
python alpaca.py cluster-up --gpu-workers 2 --cpus-per-worker 4

# Check cluster status
python alpaca.py ray-status

# Serve distributed model (auto-detects cluster)
python alpaca.py serve-ray llama2-70b

# Stop cluster when done
python alpaca.py ray-down
```

### Multi-node deployment

```bash
# On head node
python alpaca.py ray-head --dashboard-port 8265

# On worker nodes  
python alpaca.py ray-worker --address HEAD_IP:6379 --gpus 1

# Serve distributed model
python alpaca.py serve-ray llama2-70b --address ray://HEAD_IP:10001
```

### Custom API requests

```bash
# Send custom JSON payload
echo '{"model":"llama2","messages":[{"role":"user","content":"Explain quantum computing"}],"max_tokens":100}' > request.json
python alpaca.py run llama2 --json request.json

# Use a different API endpoint
python alpaca.py run llama2 --path /v1/completions -p "Once upon a time"
```

---

## 🔧 Configuration

### Environment variables

```bash
export ALPACA_CACHE_DIR="$HOME/.cache/huggingface"   # Model cache directory
export ALPACA_STATE_DIR="$HOME/.alpaca"              # State directory
export ALPACA_PORT_START=8000                        # Starting port for auto-allocation
export ALPACA_DTYPE_GPU="auto"                       # Default GPU dtype
export ALPACA_MAX_SEQS=32                            # Default max sequences
export HUGGING_FACE_HUB_TOKEN="your_token_here"      # For private models
```

### Config commands

```bash
# Show current configuration
python alpaca.py config --show

# Set configuration values
python alpaca.py config --set max_workers=4 --set timeout=300
```

---

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Alpaca CLI    │    │   vLLM Engine   │    │  Hugging Face   │
│ • Model mgmt    │────│ • Native Python │────│ • Model cache   │
│ • Process ctrl  │    │ • GPU inference │    │ • Tokenizers    │
│ • Ray clusters  │    │ • API serving   │    │ • Configs       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
          │                        │
          ▼                        ▼
┌─────────────────┐    ┌─────────────────┐
│   Ray Cluster   │    │  System Resources│
│ • Head + Workers│    │ • CUDA GPUs     │
│ • Distributed   │    │ • CPU cores     │
│ • Auto-managed  │    │ • Memory        │
└─────────────────┘    └─────────────────┘
```

---

## 🎯 Comparison

| Feature        | Alpaca         | Ollama       | vLLM CLI       | Transformers    |
| -------------- | -------------- | ------------ | -------------- | --------------- |
| Ease of Use    | ✅ Simple CLI   | ✅ Simple CLI | ❌ Complex      | ❌ Code required |
| HF Integration | ✅ Native       | ❌ Manual     | ✅ Native       | ✅ Native        |
| Native Python  | ✅ Direct       | ❌ Compiled   | ✅ Direct       | ✅ Direct        |
| Distributed    | ✅ Ray clusters | ❌ No         | ✅ Manual setup | ❌ No            |
| Process Mgmt   | ✅ Built-in     | ✅ Built-in   | ❌ Manual       | ❌ Manual        |
| Performance    | ✅ vLLM backend | ❌ llama.cpp  | ✅ vLLM         | ❌ Basic         |
| Model Formats  | ✅ HF           | ✅ GGUF       | ✅ HF           | ✅ HF            |

---

## 🐛 Troubleshooting

### Dependencies

```bash
# Check if vLLM is installed
python -c "import vllm; print('vLLM OK')"

# Check CUDA availability
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# Install missing dependencies
pip install vllm huggingface_hub httpx psutil ray
```

### Common issues

**GPU not detected**

```bash
# Check NVIDIA driver
nvidia-smi

# Check PyTorch CUDA
python -c "import torch; print(torch.cuda.is_available())"
```

**Port conflicts (address in use)**

```bash
python alpaca.py ps
python alpaca.py stop <model-name>
```

**OOM during serving**

```bash
python alpaca.py serve <model> --dtype fp16
python alpaca.py serve <model> --max-seqs 16
```

**Process management issues**

```bash
# Check running processes
python alpaca.py ps

# Kill stuck processes
python alpaca.py stop <model-name>
```

**Ray cluster issues**

```bash
# Check Ray cluster status
python alpaca.py ray-status

# Stop and restart cluster
python alpaca.py ray-down
python alpaca.py ray-head

# Check Ray installation
python -c "import ray; print('Ray OK')"
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
pip install vllm huggingface_hub httpx psutil ray
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

Alpaca combines the ease of Ollama with the power of native vLLM, making large-language-model deployment accessible to everyone. 🦙✨
