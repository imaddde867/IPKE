import json

from scripts.run_pkg_extraction import build_run_metadata


class DummyConfig:
    chunking_method = "dual_semantic"
    prompting_strategy = "P3"
    llm_backend = "llama_cpp"
    llm_model_path = "models/llm/mistral.gguf"
    llm_model_id = "mistralai/Mistral-7B-Instruct-v0.2"
    llm_quantization = "Q4_K_M"
    llm_temperature = 0.1
    llm_random_seed = 42
    gpu_backend = "cuda"


def test_build_run_metadata_contains_reproducibility_fields(tmp_path):
    doc_path = tmp_path / "doc.txt"
    doc_path.write_text("Inspect valve.", encoding="utf-8")
    metadata = build_run_metadata(
        config=DummyConfig(),
        doc_id="DOC1",
        input_path=doc_path,
        flat_path=tmp_path / "DOC1_extracted.json",
        pkg_path=tmp_path / "DOC1_pkg.json",
    )
    assert metadata["doc_id"] == "DOC1"
    assert metadata["input_sha256"]
    assert metadata["chunking_method"] == "dual_semantic"
    assert metadata["prompting_strategy"] == "P3"
    assert metadata["llm_temperature"] == 0.1
    assert metadata["llm_random_seed"] == 42
    assert metadata["git_sha"]
    json.dumps(metadata)
