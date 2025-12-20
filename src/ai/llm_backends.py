"""Backend wrappers for llama.cpp and Hugging Face transformers."""

from __future__ import annotations

import asyncio
from typing import List, Optional

from src.logging_config import get_logger

_BACKEND_ALIASES = {
    "llama.cpp": "llama_cpp",
    "llama-cpp": "llama_cpp",
}


def normalize_backend_name(name: str) -> str:
    return _BACKEND_ALIASES.get(name, name)

logger = get_logger(__name__)

try:
    from llama_cpp import Llama
    LLAMA_CPP_AVAILABLE = True
except ImportError:  # pragma: no cover
    LLAMA_CPP_AVAILABLE = False
    Llama = None

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    TRANSFORMERS_AVAILABLE = True
except ImportError:  # pragma: no cover
    torch = None
    AutoModelForCausalLM = AutoTokenizer = BitsAndBytesConfig = None
    TRANSFORMERS_AVAILABLE = False


class LLMBackendBase:
    """Synchronous text generation backend."""

    def __init__(self, config):
        self.config = config

    def generate(self, prompt: str, *, stop: Optional[List[str]] = None) -> str:
        raise NotImplementedError


class LlamaCppBackend(LLMBackendBase):
    """llama-cpp-python backend."""

    def __init__(self, config):
        super().__init__(config)
        if not LLAMA_CPP_AVAILABLE:
            raise RuntimeError("llama-cpp-python is not installed.")
        llm_config = config.get_llm_config()
        params = {
            "model_path": llm_config["model_path"],
            "n_ctx": llm_config["n_ctx"],
            "n_threads": llm_config["n_threads"],
            "n_gpu_layers": llm_config["n_gpu_layers"],
            "f16_kv": llm_config["f16_kv"],
            "use_mlock": llm_config["use_mlock"],
            "use_mmap": llm_config["use_mmap"],
            "verbose": llm_config["verbose"],
        }
        self.generation_params = {
            "max_tokens": llm_config["max_tokens"],
            "temperature": llm_config["temperature"],
            "top_p": llm_config["top_p"],
            "repeat_penalty": llm_config["repeat_penalty"],
        }
        self._model = Llama(**params)
        logger.info("Loaded llama.cpp backend with model %s", params["model_path"])

    def generate(self, prompt: str, *, stop: Optional[List[str]] = None) -> str:
        response = self._model(prompt, stop=stop or [], **self.generation_params)
        return response["choices"][0]["text"]


class TransformersBackend(LLMBackendBase):
    """Hugging Face transformers backend."""

    def __init__(self, config):
        super().__init__(config)
        if not TRANSFORMERS_AVAILABLE:
            raise RuntimeError("transformers is not installed.")

        llm_config = config.get_llm_config()
        self.device = self._resolve_device(llm_config)
        quant_config = None
        if self.device.startswith("cuda"):
            if llm_config["quantization"] == "4bit":
                quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
            elif llm_config["quantization"] == "8bit":
                quant_config = BitsAndBytesConfig(load_in_8bit=True)

        self.tokenizer = AutoTokenizer.from_pretrained(llm_config["model_id"], use_fast=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            llm_config["model_id"],
            quantization_config=quant_config,
            device_map="auto" if quant_config else None,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        if not quant_config:
            self.model.to(self.device)
        self.model.eval()
        self.generation_params = {
            "max_new_tokens": llm_config["max_tokens"],
            "temperature": llm_config["temperature"],
            "top_p": llm_config["top_p"],
            "repetition_penalty": llm_config["repeat_penalty"],
            "do_sample": True,
        }
        logger.info("Loaded transformers backend: %s", llm_config["model_id"])

    def _resolve_device(self, llm_config) -> str:
        if torch is None:
            return "cpu"
        if not llm_config.get("enable_gpu", True):
            return "cpu"
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def generate(self, prompt: str, *, stop: Optional[List[str]] = None) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        output = self.model.generate(**inputs, **self.generation_params)
        text = self.tokenizer.decode(output[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        if stop:
            for token in stop:
                idx = text.find(token)
                if idx != -1:
                    text = text[:idx]
        return text


def build_backend(config, backend_name: str) -> LLMBackendBase:
    backend = normalize_backend_name(backend_name.lower())
    if backend == "llama.cpp" or backend == "llama_cpp":
        return LlamaCppBackend(config)
    if backend == "transformers":
        return TransformersBackend(config)
    raise ValueError(f"Unsupported backend '{backend}'")


async def warmup_backend(backend: LLMBackendBase, text: str = "Hello") -> None:
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, backend.generate, text)


__all__ = [
    "LlamaCppBackend",
    "LLMBackendBase",
    "TransformersBackend",
    "build_backend",
    "normalize_backend_name",
    "warmup_backend",
]
