# 🦙 Alpaca — Ollama-style vLLM Wrapper

A streamlined CLI that brings Ollama’s simplicity to [vLLM](https://github.com/vllm-project/vllm), with native builds and distributed serving via [Ray](https://ray.io/). Build, serve, and manage large language models with ease.

> Requires Python 3.9+ ([download](https://www.python.org/downloads/)). Licensed under [MIT](https://opensource.org/licenses/MIT).

---

## ✨ Features

### 🚀 Simple model management

* **Pull & cache** models from Hugging Face
* **Serve locally** with automatic CPU/GPU detection
* **Process control**: list, stop, and manage servers
* **Aliases** for friendly model names

### 🔧 Built-in vLLM compilation

* **Source builds** for optimal performance
* **CPU optimizations** (AVX-512, Intel flags)
* **GPU builds** with CUDA detection
* **Smart dependencies** (handles PyTorch install)

### 🌐 Distributed inference

* **Ray integration** for multi-node scaling
* **Cluster orchestration** (head/worker helpers)
* **Seamless multi-GPU / multi-node** deployments

---

## 🚀 Quick Start

> In the examples below we use the `alpaca` CLI. If you’re running directly from the repo without installing, replace `alpaca` with `python alpaca.py`.

### Installation (from source)

```bash
git clone https://github.com/Slownite/alpaca.git
cd alpaca
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
pip install -r requirements.txt
# Optional (recommended for CLI entry point):
pip install -e .
```

### Build vLLM (first time)

```bash
# Auto-detect hardware and build vLLM
alpaca build-vllm --device auto

# Or specify explicitly
alpaca build-vllm --device cpu    # CPU-only
alpaca build-vllm --device gpu    # GPU build
```

### Serve your first model

```bash
# Pull a model from Hugging Face and alias it
alpaca pull microsoft/DialoGPT-medium --alias chatbot

# Start serving
alpaca serve chatbot

# Test a prompt
alpaca run chatbot -p "Hello, how are you today?"
```

---

## 📋 Command Reference

### 🔧 vLLM Build System

| Command        | Description            | Example                          |
| -------------- | ---------------------- | -------------------------------- |
| `build-vllm`   | Build vLLM from source | `alpaca build-vllm --device cpu` |
| `build-status` | Show build information | `alpaca build-status`            |
| `build-clean`  | Clean build artifacts  | `alpaca build-clean --all`       |

**Build options**

* `--device {cpu,gpu,auto}` target device (default: auto)
* `--ref BRANCH` git branch/tag (default: `main`)
* `--wheel` build a wheel instead of editable install
* `--no-test` skip installation test
* `--force-rebuild` rebuild even if already built

### 🦙 Model Management

| Command | Description            | Example                                 |
| ------- | ---------------------- | --------------------------------------- |
| `pull`  | Download model from HF | `alpaca pull meta-llama/Llama-2-7b-hf`  |
| `ls`    | List cached models     | `alpaca ls --size`                      |
| `serve` | Start model server     | `alpaca serve llama2 --port 8000`       |
| `ps`    | List running servers   | `alpaca ps`                             |
| `stop`  | Stop a server          | `alpaca stop llama2`                    |
| `rm`    | Remove server/alias    | `alpaca rm llama2 --purge`              |
| `run`   | Send test request      | `alpaca run llama2 -p "Tell me a joke"` |
| `logs`  | Show server logs       | `alpaca logs llama2`                    |

**Serve options**

* `--gpu` / `--cpu` force mode
* `--port PORT` specify port (default: auto)
* `--dtype {auto,float32,bf16,fp16}` precision
* `--max-seqs N` max concurrent sequences

### 🌐 Distributed (Ray)

| Command        | Description             | Example                                              |
| -------------- | ----------------------- | ---------------------------------------------------- |
| `cluster-up`   | Start local Ray cluster | `alpaca cluster-up --gpu-workers 2`                  |
| `serve-ray`    | Serve with Ray backend  | `alpaca serve-ray llama2 --address ray://head:10001` |
| `cluster-down` | Stop Ray cluster        | `alpaca cluster-down`                                |
| `ray-head`     | Start Ray head node     | `alpaca ray-head --dashboard-port 8265`              |
| `ray-worker`   | Start Ray worker        | `alpaca ray-worker --address head:6379 --gpu`        |

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
export ALPACA_CACHE_DIR="$HOME/.cache/huggingface"   # Model cache
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
│   Alpaca CLI    │    │   vLLM Engine   │    │  Hugging Face   │
│ • Model mgmt    │────│ • Inference     │────│ • Model storage │
│ • Process ctrl  │    │ • Serving       │    │ • Tokenizers    │
│ • Build system  │    │ • Optimization  │    │ • Configs       │
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
| Source Builds  | ✅ Built-in     | ❌ No         | ❌ Manual       | ❌ Manual        |
| Distributed    | ✅ Ray support  | ❌ No         | ✅ Manual setup | ❌ No            |
| Process Mgmt   | ✅ Built-in     | ✅ Built-in   | ❌ Manual       | ❌ Manual        |
| Performance    | ✅ vLLM backend | ❌ llama.cpp  | ✅ vLLM         | ❌ Basic         |
| Model Formats  | ✅ HF/GGUF      | ✅ GGUF       | ✅ HF           | ✅ HF            |

---

## 🐛 Troubleshooting

### Build issues

```bash
# Python 3.13 compatibility:
# Alpaca will use PyTorch nightly automatically under Python 3.13+

# Clean and rebuild
alpaca build-clean --all
alpaca build-vllm --device cpu --force-rebuild

# Check build status
alpaca build-status
```

### Common issues

**ImportError: vLLM not found**

```bash
alpaca build-vllm --device auto
```

**Port conflicts (address in use)**

```bash
alpaca ps
alpaca stop <model-name>
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
pip install -r requirements.txt
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
