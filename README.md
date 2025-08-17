⏺ 🦙 Alpaca - Ollama-style vLLM Wrapper

  https://www.python.org/downloads/
  https://github.com/vllm-project/vllm
  https://ray.io/
  https://opensource.org/licenses/MIT

  A streamlined CLI tool that brings Ollama's simplicity to vLLM with native execution and distributed Ray support. Build, serve, and manage large language models with
  ease.

  ✨ Features

  🚀 Simple Model Management

  - Pull & Cache: Download models from Hugging Face with automatic caching
  - Serve Locally: Start vLLM servers with automatic CPU/GPU detection
  - Process Control: List, stop, and manage running model servers
  - Aliases: Create friendly names for your favorite models

  🔧 Built-in vLLM Compilation

  - Source Builds: Compile vLLM from source for optimal performance
  - CPU Optimization: Automatic AVX-512 detection and Intel optimizations
  - GPU Support: CUDA detection and GPU-optimized builds
  - Smart Dependencies: Handles PyTorch installation for your platform

  🌐 Distributed Inference

  - Ray Integration: Scale across multiple nodes with Ray distributed execution
  - Cluster Management: Built-in Ray head/worker orchestration
  - Multi-Node: Deploy models across GPU clusters seamlessly

  🚀 Quick Start

  Installation

  git clone https://github.com/yourusername/alpaca.git
  cd alpaca
  python -m venv venv
  source venv/bin/activate  # On Windows: venv\Scripts\activate
  pip install -r requirements.txt

  Build vLLM (First Time)

  # Auto-detect your hardware and build vLLM
  python alpaca.py build-vllm --device auto

  # Or specify CPU/GPU explicitly
  python alpaca.py build-vllm --device cpu    # For CPU-only builds
  python alpaca.py build-vllm --device gpu    # For GPU builds

  Serve Your First Model

  # Pull a model from Hugging Face
  python alpaca.py pull microsoft/DialoGPT-medium --alias chatbot

  # Start serving the model
  python alpaca.py serve chatbot

  # Test it out
  python alpaca.py run chatbot -p "Hello, how are you today?"

  📋 Command Reference

  🔧 vLLM Build System

  | Command      | Description            | Example                        |
  |--------------|------------------------|--------------------------------|
  | build-vllm   | Build vLLM from source | alpaca build-vllm --device cpu |
  | build-status | Show build information | alpaca build-status            |
  | build-clean  | Clean build artifacts  | alpaca build-clean --all       |

  Build Options:
  - --device {cpu,gpu,auto} - Target device (default: auto-detect)
  - --ref BRANCH - Git branch/tag to build (default: main)
  - --wheel - Build wheel instead of editable install
  - --no-test - Skip installation test
  - --force-rebuild - Force rebuild even if already built

  🦙 Model Management

  | Command | Description            | Example                               |
  |---------|------------------------|---------------------------------------|
  | pull    | Download model from HF | alpaca pull meta-llama/Llama-2-7b-hf  |
  | ls      | List cached models     | alpaca ls --size                      |
  | serve   | Start model server     | alpaca serve llama2 --port 8000       |
  | ps      | List running servers   | alpaca ps                             |
  | stop    | Stop a server          | alpaca stop llama2                    |
  | rm      | Remove server/alias    | alpaca rm llama2 --purge              |
  | run     | Send test request      | alpaca run llama2 -p "Tell me a joke" |
  | logs    | Show server logs       | alpaca logs llama2                    |

  Serve Options:
  - --gpu / --cpu - Force GPU/CPU mode
  - --port PORT - Specify port (default: auto)
  - --dtype {auto,float32,bf16,fp16} - Model precision
  - --max-seqs N - Max concurrent sequences

  🌐 Distributed Ray

  | Command      | Description             | Example                                            |
  |--------------|-------------------------|----------------------------------------------------|
  | cluster-up   | Start local Ray cluster | alpaca cluster-up --gpu-workers 2                  |
  | serve-ray    | Serve with Ray backend  | alpaca serve-ray llama2 --address ray://head:10001 |
  | cluster-down | Stop Ray cluster        | alpaca cluster-down                                |
  | ray-head     | Start Ray head node     | alpaca ray-head --dashboard-port 8265              |
  | ray-worker   | Start Ray worker        | alpaca ray-worker --address head:6379 --gpu        |

  📖 Usage Examples

  Local Development

  # Pull and serve a small model for testing
  alpaca pull facebook/opt-125m --alias tiny
  alpaca serve tiny --cpu
  alpaca run tiny -p "The future of AI is"

  Production Serving

  # Serve a larger model with GPU acceleration
  alpaca pull meta-llama/Llama-2-7b-chat-hf --alias llama2-chat
  alpaca serve llama2-chat --gpu --dtype bf16 --max-seqs 64

  Multi-Node Deployment

  # On head node
  alpaca ray-head --dashboard-port 8265

  # On worker nodes  
  alpaca ray-worker --address HEAD_IP:6379 --gpu

  # Serve distributed model
  alpaca serve-ray llama2-70b --address ray://HEAD_IP:10001

  Custom API Requests

  # Send custom JSON payload
  echo '{"model":"llama2","messages":[{"role":"user","content":"Explain quantum computing"}],"max_tokens":100}' > request.json
  alpaca run llama2 --json request.json

  # Use different API endpoint
  alpaca run llama2 --path /v1/completions -p "Once upon a time"

  🔧 Configuration

  Environment Variables

  export ALPACA_CACHE_DIR="$HOME/.cache/huggingface"    # Model cache location
  export ALPACA_STATE_DIR="$HOME/.alpaca"               # Alpaca state directory
  export ALPACA_PORT_START=8000                         # Starting port for auto-allocation
  export ALPACA_DTYPE_CPU="float32"                     # Default CPU dtype
  export ALPACA_DTYPE_GPU="auto"                        # Default GPU dtype
  export ALPACA_MAX_SEQS=32                             # Default max sequences
  export HUGGING_FACE_HUB_TOKEN="your_token_here"      # For private models

  Config Commands

  # Show current configuration
  alpaca config --show

  # Set configuration values
  alpaca config --set max_workers=4 --set timeout=300

  🏗️ Architecture

  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
  │   Alpaca CLI    │    │  vLLM Engine    │    │ Hugging Face    │
  │                 │────│                 │────│     Hub         │
  │ • Model mgmt    │    │ • Inference     │    │ • Model storage │
  │ • Process ctrl  │    │ • Serving       │    │ • Tokenizers    │
  │ • Build system  │    │ • Optimization  │    │ • Configs       │
  └─────────────────┘    └─────────────────┘    └─────────────────┘
           │                       │
           │                       │
           ▼                       ▼
  ┌─────────────────┐    ┌─────────────────┐
  │  Ray Cluster    │    │ Local Hardware  │
  │                 │    │                 │
  │ • Multi-node    │    │ • CPU/GPU       │
  │ • Load balance  │    │ • Memory        │
  │ • Fault tolerance│   │ • Storage       │
  └─────────────────┘    └─────────────────┘

  🎯 Comparison

  | Feature        | Alpaca         | Ollama       | vLLM CLI       | Transformers    |
  |----------------|----------------|--------------|----------------|-----------------|
  | Ease of Use    | ✅ Simple CLI   | ✅ Simple CLI | ❌ Complex      | ❌ Code required |
  | HF Integration | ✅ Native       | ❌ Manual     | ✅ Native       | ✅ Native        |
  | Source Builds  | ✅ Built-in     | ❌ No         | ❌ Manual       | ❌ Manual        |
  | Distributed    | ✅ Ray support  | ❌ No         | ✅ Manual setup | ❌ No            |
  | Process Mgmt   | ✅ Built-in     | ✅ Built-in   | ❌ Manual       | ❌ Manual        |
  | Performance    | ✅ vLLM backend | ❌ llama.cpp  | ✅ vLLM         | ❌ Basic         |
  | Model Formats  | ✅ HF/GGUF      | ✅ GGUF       | ✅ HF           | ✅ HF            |

  🐛 Troubleshooting

  Build Issues

  # Python 3.13 compatibility
  # Alpaca automatically uses PyTorch nightly for Python 3.13+

  # Clean and rebuild
  alpaca build-clean --all
  alpaca build-vllm --device cpu --force-rebuild

  # Check build status
  alpaca build-status

  Common Issues

  Import Error: vLLM not found
  # Rebuild vLLM
  alpaca build-vllm --device auto

  Port Conflicts: Address already in use
  # Check running processes
  alpaca ps

  # Stop conflicting server
  alpaca stop model-name

  Memory Issues: OOM during serving
  # Reduce precision
  alpaca serve model --dtype fp16

  # Reduce max sequences
  alpaca serve model --max-seqs 16

  🤝 Contributing

  1. Fork the repository
  2. Create a feature branch (git checkout -b feature/amazing-feature)
  3. Commit your changes (git commit -m 'Add amazing feature')
  4. Push to the branch (git push origin feature/amazing-feature)
  5. Open a Pull Request

  Development Setup

  git clone https://github.com/yourusername/alpaca.git
  cd alpaca
  python -m venv dev-env
  source dev-env/bin/activate
  pip install -r requirements.txt
  pip install -e .

  📄 License

  This project is licensed under the MIT License - see the LICENSE file for details.

  🙏 Acknowledgments

  - https://github.com/vllm-project/vllm - High-performance LLM inference engine
  - https://ray.io/ - Distributed computing framework
  - https://ollama.ai/ - Inspiration for the simple CLI interface
  - https://huggingface.co/ - Model hub and transformers library

  📞 Support

  - Issues: https://github.com/yourusername/alpaca/issues
  - Discussions: https://github.com/yourusername/alpaca/discussions
  - Documentation: https://github.com/yourusername/alpaca/wiki

  ---
  Alpaca combines the ease of Ollama with the power and flexibility of vLLM, making large language model deployment accessible to everyone. 🦙✨

