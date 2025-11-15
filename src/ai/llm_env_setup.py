"""Set LLM environment variables before any heavy imports."""
import os

# Must be set before tokenizers library loads
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
