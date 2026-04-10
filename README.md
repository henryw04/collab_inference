# Guide to run the pipeline

1. Follow the [Setup Guide](#️-environment-setup-guide) Below

2. Make Sure the config files (eg API_config, controller_config, worker_config) are correctly setup as you may want different inference. Check them once before proceeding. Most instructions are written inside the config.

3. Start the pipeline with (make sure venv is activate):

**Starting API**
```bash
uvicorn API:app
```

**Starting Controller**
```bash
python controller.py
``` 

**Starting Worker**
```bash
python worker.py
``` 

4. Enjoy the pipeline!

# 🛠️ Environment Setup Guide

Follow these steps to create an isolated Python environment and install all dependencies for the `collab_inference` project.

## 📦 1. Create Virtual Environment
Open your terminal or command prompt in the project root directory (`collab_inference/`) and run:

```bash
python -m venv .venv
```
> 💡 This creates a new folder named `.venv` containing an isolated Python environment.

## 🔌 2. Activate Virtual Environment
Activation commands differ by operating system:

**Windows (CMD / PowerShell):**
```cmd
.venv\Scripts\activate
```

**macOS / Linux:**
```bash
source .venv/bin/activate
```
> ✅ You'll know it's active when your terminal prompt shows `(.venv)` at the beginning.

## 📥 3. Install Dependencies
Once activated, upgrade `pip` and install all required packages:

```bash
python -m pip install --upgrade pip
pip install -r req.txt
```

## 🚀 Ready to Run
Your environment is now ready! Proceed to run the module:
```bash
uvicorn API:app
python controller.py
python worker.py
```

## ⚠️ Troubleshooting
- **`python` not found?** Try `python3` instead.
- **Permission errors?** Run terminal as Administrator (Windows) or use `sudo` (Linux/macOS) only if absolutely necessary.
- **GPU Support?** If you need CUDA, install PyTorch with GPU support:
  ```bash
  pip install torch --index-url https://download.pytorch.org/whl/cu118
  ```
- **Exit environment later:** Simply run `deactivate` in your terminal.
