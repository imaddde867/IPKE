from src.ai.llm_backends import normalize_backend_name


def test_normalize_backend_name():
    assert normalize_backend_name("llama.cpp") == "llama_cpp"
    assert normalize_backend_name("llama-cpp") == "llama_cpp"
    assert normalize_backend_name("llama_cpp") == "llama_cpp"
    assert normalize_backend_name("transformers") == "transformers"


def test_llama_backend_passes_seed_to_llama():
    """LlamaCppBackend must pass random_seed from config to Llama(seed=...)."""
    from unittest.mock import MagicMock, patch
    import src.ai.llm_backends as lb

    mock_llama_cls = MagicMock()
    mock_llama_cls.return_value = MagicMock()

    mock_config = MagicMock()
    mock_config.get_llm_config.return_value = {
        "model_path": "/tmp/fake.gguf",
        "n_ctx": 512,
        "n_threads": 4,
        "n_gpu_layers": 0,
        "f16_kv": True,
        "use_mlock": False,
        "use_mmap": True,
        "verbose": False,
        "random_seed": 42,
        "max_tokens": 128,
        "temperature": 0.1,
        "top_p": 0.9,
        "repeat_penalty": 1.1,
    }

    # patch at module level — LlamaCppBackend.__init__ looks up `Llama` as a
    # global in src.ai.llm_backends, so patching the module-level name works.
    with patch("src.ai.llm_backends.Llama", mock_llama_cls), \
         patch("src.ai.llm_backends.LLAMA_CPP_AVAILABLE", True):
        lb.LlamaCppBackend(mock_config)

    call_kwargs = mock_llama_cls.call_args[1]
    assert "seed" in call_kwargs, (
        f"Llama() must be called with seed=, got kwargs: {call_kwargs}"
    )
    assert call_kwargs["seed"] == 42, (
        f"Expected seed=42, got seed={call_kwargs['seed']}"
    )
