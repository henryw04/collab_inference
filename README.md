# collab_inference 🤝

A distributed inference framework for running large language models (like Llama) across multiple devices or processes by splitting model layers. Built with **FastAPI**, **ZeroMQ**, and **Hugging Face Transformers**.

> 🎯 **Core Idea**: Instead of loading an entire LLM on one machine, partition its transformer layers across multiple workers. Each worker processes only its assigned layers, passing hidden states through a ZeroMQ pipeline. This enables inference on resource-constrained edge devices or heterogeneous hardware setups.

---

## 🏗️ Architecture Overview

```
┌─────────────┐      ┌─────────────┐      ┌──────────────┐      ┌─────────────┐
│   Client    │----->│    API      │----->│ Controller   │----->│  Worker 1   │
│  (HTTP POST)│      │ (FastAPI)   │      │(Orchestrator)│      │(Layers 0-7) │
└─────────────┘      └─────────────┘      └──────────────┘      └──────┬──────┘
                                                                       │
                                                                       ▼
┌─────────────┐     ┌─────────────┐     ┌─────────────────┐      ┌─────────────┐
│   Response  │<----│    API      │<----│ Controller      │<-----│  Worker N   │
│  (JSON)     │     │ (FastAPI)   │     │(Logits/Decoding)│      │(Layers X-Y) │
└─────────────┘     └─────────────┘     └─────────────────┘      └─────────────┘
```

### Component Roles

| Component | Responsibility | Key Technologies |
|-----------|---------------|------------------|
| **API** (`API.py`) | HTTP endpoint (`POST /ask`), request timing, ZeroMQ client | FastAPI, `zmq.asyncio` |
| **Controller** (`controller.py`) | Tokenization, embedding lookup, logits processing, generation loop, worker coordination | Transformers, ZeroMQ ROUTER/PUSH/PULL |
| **Worker** (`worker.py`) | Execute assigned transformer layers, forward hidden states | PyTorch, ZeroMQ PUSH/PULL |
| **Config** (`config/`) | Network endpoints, layer partitioning, inference hyperparameters | Python dataclasses |

---

## 🚀 Quick Start

### 1. Environment Setup

Clone and enter project
```bash
git clone https://github.com/henryw04/collab_inference.git
cd collab_inference
```
Create virtual environment
```bash
python -m venv .venv
```

Activate

**Windows (CMD / PowerShell):**
```cmd
.venv\Scripts\activate
```

**macOS / Linux:**
```bash
source .venv/bin/activate
```
Upgrade pip and install dependencies
```bash
python -m pip install --upgrade pip
pip install "fastapi[standard]" pyzmq "transformers[torch]" accelerate psutil
```

> 💡 **GPU users**: If you need CUDA support, install PyTorch with the appropriate backend:
> ```bash
> pip install torch --index-url https://download.pytorch.org/whl/cu118  # Adjust CUDA version as needed
> ```

### 2. Configure Your Pipeline

Edit the files in `config/` before launching:

- **`config/API_config.py`**: ZeroMQ endpoint for API↔Controller communication
- **`config/controller_config.py`**: Controller's bind/connect addresses for API and workers
- **`config/worker_config.py`**: 
  - `LAYER_RANGES`: Which transformer layers this worker executes (e.g., `(0, 8)` for layers 0–7)
  - `PULL_IP` / `PUSH_IP`: ZeroMQ addresses for chaining workers
  - `WORKER_ID`: Unique identifier (used for logging and routing)

> 🔗 **Multi-machine tip**: Replace `127.0.0.1` with actual IPs when deploying across devices. Use `tcp://*:PORT` for `PUSH_IP` to accept connections from any interface.

### 3. Launch the Pipeline (in separate terminals)

```bash
# Terminal 1: API server
uvicorn API:app --host 127.0.0.1 --port 8000

# Terminal 2: Controller
python controller.py

# Terminal 3+: Workers (one per device/layer partition)
python worker.py
```

### 4. Test It

```bash
curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"input": "Explain how distributed inference works"}'
```

Response includes the generated text plus timing metrics:
```json
{
  "result": "Distributed inference splits model computation...",
  "start_time": 1712847234.12,
  "end_time": 1712847236.45,
  "total_time": 2.33
}
```

---

## ⚙️ Configuration Deep Dive

### Layer Partitioning Strategy

The model (e.g., `openlm-research/open_llama_3b`) is split by transformer layer index. For a 26-layer model:

```python
# Worker 1: embedding + layers 0-7
LAYER_RANGES = (0, 8)

# Worker 2: layers 8-15  
LAYER_RANGES = (8, 16)

# Worker 3: layers 16-25 + final norm/lm_head (handled by Controller)
LAYER_RANGES = (16, 26)
```

> 📐 **Memory planning**: Use `benchmark.py` to estimate how many layers fit on a device given its available RAM and target precision (FP16/FP32).

### Inference Hyperparameters (`config/common.py`)

| Parameter | Default | Effect |
|-----------|---------|--------|
| `MAX_NEW_TOKENS` | 20 | Max tokens to generate per request |
| `TEMPERATURE` | 0.1 | Sampling randomness (lower = more deterministic) |
| `TOP_P` | 0.9 | Nucleus sampling threshold |
| `REPETITION_PENALTY` | 1.2 | Discourage token repetition |
| `MODEL_NAME` | `openlm-research/open_llama_3b` | Hugging Face model identifier |

All values can be overridden per-request by extending the `State` class.

---

## 🧪 Utilities

### Baseline Comparison (`baseline.py`)
Run single-device inference for performance benchmarking:
```bash
python baseline.py
```
Outputs generation time, tokens/sec, and saves results to `baseline_result.txt`.

### Layer Capacity Estimator (`benchmark.py`)
Interactive tool to calculate max layers per device based on RAM:
```bash
python benchmark.py
# Follow prompts for model size (e.g., 3 for 3B) and precision (16 for FP16)
```

---

## 🔍 Troubleshooting

| Issue | Likely Cause | Solution |
|-------|-------------|----------|
| `Connection refused` on ZeroMQ ports | Workers not started in order, or firewall blocking | Start Controller first, then Worker 1→N; ensure ports are open |
| `CUDA out of memory` | Too many layers assigned to one GPU | Reduce `LAYER_RANGES` span; use `benchmark.py` to recalculate |
| Slow token generation | Network latency between workers | Co-locate tightly-coupled workers; use `tcp://127.0.0.1` for local testing |
| `python` command not found | System uses `python3` | Replace `python` with `python3` in setup steps |
| Model download fails | HF cache permissions | The project auto-sets `HF_HOME` to `./hf_cache`; ensure write access |

To exit the virtual environment later:
```bash
deactivate
```

---

## 📁 Project Structure

```
collab_inference/
├── API.py                 # FastAPI endpoint + ZeroMQ client
├── controller.py          # Generation loop, tokenization, worker orchestration
├── worker.py              # Layer execution module
├── baseline.py            # Single-device reference implementation
├── benchmark.py           # RAM-based layer capacity calculator
├── config/
│   ├── __init__.py        # Config exports
│   ├── common.py          # Shared constants, State class, seeding
│   ├── API_config.py      # API↔Controller ZeroMQ address
│   ├── controller_config.py # Controller socket bindings
│   └── worker_config.py   # Layer ranges, worker networking
└── hf_cache/              # Auto-created Hugging Face model cache
```

---

## 🤝 Contributing & Future Work

This is an active research prototype. Potential extensions:
- [ ] Dynamic layer rebalancing based on device load
- [ ] Support for pipeline parallelism with attention KV-cache handoff
- [ ] Integration with ONNX Runtime or TensorRT for worker acceleration
- [ ] Web UI for monitoring pipeline latency per stage

Found a bug or have an idea? Open an issue or PR — all contributions welcome.

---

> ℹ️ **Note**: This project is designed for research and educational purposes. For production deployments, consider additional safeguards like request authentication, rate limiting, and graceful degradation handling.

*Built with ❤️ for edge AI and collaborative computing.* 🌐✨
