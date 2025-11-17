## Project Overview

This project, named "Industrial Procedural Knowledge Extraction (IPKE)," is a Python-based system designed to extract structured procedural knowledge from unstructured technical documents. It leverages a local Large Language Model (LLM), specifically Mistral-7B Instruct, with a dual-backend architecture supporting both Metal for Apple Silicon and CUDA for NVIDIA GPUs.

The system features a multi-GPU setup for enhanced throughput, various prompt engineering strategies, and a suite of semantic chunking methods. It provides a FastAPI for programmatic access and a Streamlit application for an interactive user interface. The project is well-structured, with separate directories for source code, tests, configurations, and data.

## Building and Running

### 1. Environment Setup

- Python 3.10+
- For Metal: macOS with Apple Silicon
- For CUDA: Linux/Windows with an NVIDIA GPU

### 2. Install Dependencies

```bash
python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 3. Model Setup

**For `metal` backend (Apple Silicon):**

Download the GGUF model manually:

```bash
huggingface-cli login
huggingface-cli download TheBloke/Mistral-7B-Instruct-v0.2-GGUF \
  --include "mistral-7b-instruct-v0.2.Q4_K_M.gguf" \
  --local-dir models/llm --local-dir-use-symlinks False
```

**For `cuda` backend (NVIDIA GPU):**

The `transformers` library will automatically download the model on the first run.

### 4. Configuration

Copy the `.env.example` file to `.env` and set the `GPU_BACKEND` to `metal` or `cuda`.

### 5. Running the Application

**API:**

```bash
python main.py
```

**UI:**

```bash
streamlit run streamlit_app.py
```

### 6. Testing

To run the test suite:

```bash
pytest
```

## Development Conventions

- **Configuration:** The project uses a centralized configuration system in `src/core/unified_config.py`, which loads settings from environment variables and a `.env` file.
- **Logging:** The application uses JSON logging with correlation IDs for request tracing.
- **Testing:** The project includes a `tests/` directory with unit and integration tests. The `pytest` framework is used for testing.
- **API:** A FastAPI application provides the core API, with routes defined in `src/api/app.py`.
- **UI:** A Streamlit application in `streamlit_app.py` provides a user-friendly interface for document extraction.
- **Code Style:** The code appears to follow standard Python conventions (PEP 8).
