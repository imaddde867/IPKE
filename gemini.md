# Industrial Procedural Knowledge Extraction (IPKE)

This document provides a comprehensive overview of the IPKE project, its architecture, and instructions for building, running, and configuring the system.

## Project Overview

The Industrial Procedural Knowledge Extraction (IPKE) system is a Python-based application designed to extract structured procedural knowledge from unstructured technical documents. It leverages a local Large Language Model (LLM), specifically Mistral-7B-Instruct, to perform the extraction.

The system features a dual-backend architecture for LLM inference, supporting both Apple Silicon (Metal) and NVIDIA (CUDA) GPUs. It includes a FastAPI backend for serving the extraction API and a Streamlit frontend for interactive use.

### Key Technologies

*   **Backend:** Python, FastAPI
*   **Frontend:** Streamlit
*   **LLM:** Mistral-7B-Instruct
*   **LLM Backends:** `llama.cpp` (for Metal), `transformers` (for CUDA)
*   **Data Handling:** Pydantic, Pandas
*   **Configuration:** Unified configuration system with `.env` support

### Architecture

The project is structured as follows:

*   `main.py`: The entry point for the FastAPI server.
*   `streamlit_app.py`: The entry point for the Streamlit UI.
*   `src/api/app.py`: Defines the FastAPI application, routes, and middleware.
*   `src/ai/knowledge_engine.py`: Implements the dual-backend LLM integration.
*   `src/processors/streamlined_processor.py`: Orchestrates document loading, chunking, and knowledge extraction.
*   `src/core/unified_config.py`: Provides a single source of truth for all configuration settings.
*   `tools/evaluate.py`: A tool for evaluating structured predictions.
*   `scripts/`: Contains various scripts for running baselines, plotting metrics, and other tasks.

## Building and Running

### 1. Installation

1.  **Create and activate a virtual environment:**

    ```bash
    python -m venv .venv
    source .venv/bin/activate
    ```

2.  **Install dependencies:**

    ```bash
    pip install --upgrade pip
    pip install -r requirements.txt
    ```

3.  **Install the spaCy English model:**

    ```bash
    python -m spacy download en_core_web_sm
    ```

### 2. Model Setup

The system requires a local LLM. The setup depends on your GPU backend.

*   **For Metal (Apple Silicon):** Download the GGUF model manually.

    ```bash
    huggingface-cli login
    huggingface-cli download TheBloke/Mistral-7B-Instruct-v0.2-GGUF \
      --include "mistral-7b-instruct-v0.2.Q4_K_M.gguf" \
      --local-dir models/llm --local-dir-use-symlinks False
    ```

*   **For CUDA (NVIDIA GPU):** The `transformers` library will automatically download the model on the first run.

### 3. Configuration

1.  Copy the `.env.example` file to `.env`:

    ```bash
    cp .env.example .env
    ```

2.  Edit the `.env` file to set the `GPU_BACKEND` (`metal` or `cuda`) and other configuration options as needed.

### 4. Running the Application

*   **Run the API server:**

    ```bash
    python main.py
    ```

    The API documentation will be available at `http://localhost:8000/docs`.

*   **Run the Streamlit UI:**

    ```bash
    streamlit run streamlit_app.py
    ```

    The UI will be available at `http://localhost:8501`.

### 5. Running Evaluations

The project includes scripts for running baseline evaluations and plotting metrics.

*   **Run preflight checks:**

    ```bash
    python scripts/baseline_preflight.py
    ```

*   **Run extraction and evaluation loops:**

    ```bash
    python scripts/run_baseline_loops.py --runs 3 --out logs/baseline_runs
    ```

*   **Plot metrics:**

    ```bash
    python scripts/plot_baseline_metrics.py
    ```

## Development Conventions

### Configuration

All configuration for the application is managed through the `src/core/unified_config.py` module. This module loads settings from environment variables and `.env` files, with different defaults for development, testing, and production environments.

To modify the configuration, you can either set environment variables or edit the `.env` file.

### Code Structure

The main application logic is located in the `src/` directory. The code is organized into modules for the API, AI components, core functionality, and processors.

### Testing

Tests are located in the `tests/` directory. The project uses `pytest` for running tests.
