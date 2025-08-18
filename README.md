# 🦙 Alpaca — Ollama-style vLLM Wrapper

A streamlined CLI that brings Ollama's simplicity to [vLLM](https://github.com/vllm-project/vllm) with native Python processes and optional distributed serving via [Ray](https://ray.io/). Pull, serve, and manage large language models with ease.

> Requires CUDA GPU, Python 3.9+, vLLM, and optional dependencies. Licensed under [MIT](https://opensource.org/licenses/MIT).

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

### 🌐 Distributed inference

* **Ray integration** for multi-node scaling
* **Manual cluster setup** (requires pre-existing Ray cluster)
* **Distributed serving** for large models

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
pip install vllm huggingface_hub httpx psutil
```

#### Option 2: Direct dependencies

```bash
pip install vllm huggingface_hub httpx psutil
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

### 🌐 Distributed (Ray)

| Command        | Description                           | Example                                                           |
| -------------- | ------------------------------------- | ----------------------------------------------------------------- |
| `serve-ray`    | Serve with Ray backend (requires pre-existing Ray cluster) | `python alpaca.py serve-ray llama2 --address ray://head:10001` |

**Note**: Ray cluster management is manual. You need to set up Ray cluster using `ray start --head` and `ray start --address=<head-ip>:6379` on worker nodes.

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

### Multi-node deployment

```bash
# On head node
ray start --head --dashboard-port=8265

# On worker nodes  
ray start --address=HEAD_IP:6379

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
│   Alpaca CLI    │    │   vLLM Native   │    │  Hugging Face   │
│ • Model mgmt    │────│ • Python procs  │────│ • Model storage │
│ • Process ctrl  │    │ • GPU serving   │    │ • Tokenizers    │
│ • Ray orchestr. │    │ • Port mgmt     │    │ • Configs       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
          │                        │
          ▼                        ▼
┌─────────────────┐    ┌─────────────────┐
│   Ray Cluster   │    │  Local Hardware │
│ • Multi-node    │    │ • CUDA GPU      │
│ • Distributed   │    │ • Memory        │
│ • (Optional)    │    │ • Storage       │
└─────────────────┘    └─────────────────┘
```

---

## 🎯 Comparison

| Feature        | Alpaca         | Ollama       | vLLM CLI       | Transformers    |
| -------------- | -------------- | ------------ | -------------- | --------------- |
| Ease of Use    | ✅ Simple CLI   | ✅ Simple CLI | ❌ Complex      | ❌ Code required |
| HF Integration | ✅ Native       | ❌ Manual     | ✅ Native       | ✅ Native        |
| Native Python  | ✅ Direct       | ❌ Compiled   | ✅ Direct       | ✅ Direct        |
| Distributed    | ✅ Ray support  | ❌ No         | ✅ Manual setup | ❌ No            |
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
pip install vllm huggingface_hub httpx psutil
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
pip install vllm huggingface_hub httpx psutil
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
