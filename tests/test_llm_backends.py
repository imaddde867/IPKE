from src.ai.llm_backends import normalize_backend_name


def test_normalize_backend_name():
    assert normalize_backend_name("llama.cpp") == "llama_cpp"
    assert normalize_backend_name("llama-cpp") == "llama_cpp"
    assert normalize_backend_name("llama_cpp") == "llama_cpp"
    assert normalize_backend_name("transformers") == "transformers"
