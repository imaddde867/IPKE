import asyncio
import os
import tempfile
from pathlib import Path
from typing import Dict, Any, List

import pandas as pd
import streamlit as st

from src.core.unified_config import get_config
from src.processors.streamlined_processor import StreamlinedDocumentProcessor
from src.exceptions import ProcessingError

st.set_page_config(
    page_title="Industrial Procedural Knowledge Extraction (IPKE)",
    page_icon="🐴",
    layout="wide",
)


def _ensure_session_defaults() -> None:
    config = get_config()
    defaults = {
        "quality_threshold": config.quality_threshold,
        "confidence_threshold": config.confidence_threshold,
        "chunk_size": config.chunk_size,
        "llm_n_ctx": config.llm_n_ctx,
        "llm_temperature": config.llm_temperature,
        "llm_top_p": config.llm_top_p,
        "llm_repeat_penalty": config.llm_repeat_penalty,
        "llm_max_tokens": config.llm_max_tokens,
        "llm_n_threads": config.llm_n_threads,
        "max_workers": config.max_workers,
        "enable_gpu": config.enable_gpu,
        "gpu_backend": config.gpu_backend,
        "llm_n_gpu_layers": config.llm_n_gpu_layers,
        "gpu_memory_fraction": config.gpu_memory_fraction,
    }
    if "settings" not in st.session_state:
        st.session_state["settings"] = defaults.copy()
    else:
        for key, value in defaults.items():
            st.session_state["settings"].setdefault(key, value)

    st.session_state.setdefault("processor", None)
    st.session_state.setdefault("last_result", None)


def _apply_settings(new_values: Dict[str, Any]) -> None:
    config = get_config()
    for key, value in new_values.items():
        setattr(config, key, value)
    st.session_state["settings"] = new_values.copy()
    st.session_state["processor"] = None  # rebuild with fresh config


def _get_processor() -> StreamlinedDocumentProcessor:
    processor = st.session_state.get("processor")
    if processor is None:
        processor = StreamlinedDocumentProcessor()
        st.session_state["processor"] = processor
    return processor


def _process_entities(entities) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for entity in entities:
        rows.append(
            {
                "Content": entity.content,
                "Type": entity.entity_type,
                "Category": entity.category,
                "Confidence": round(entity.confidence, 3),
                "Context": entity.context,
            }
        )
    return pd.DataFrame(rows)


def _render_sidebar() -> None:
    config = get_config()
    current = st.session_state["settings"]

    with st.sidebar.form("settings_form"):
        st.subheader("Extraction Settings")
        confidence_threshold = st.slider(
            "Confidence Threshold",
            min_value=0.0,
            max_value=1.0,
            step=0.01,
            value=float(current["confidence_threshold"]),
        )
        quality_threshold = st.slider(
            "Quality Threshold",
            min_value=0.0,
            max_value=1.0,
            step=0.01,
            value=float(current["quality_threshold"]),
        )
        st.subheader("Throughput")
        col_left, col_right = st.columns(2)
        with col_left:
            chunk_size = st.number_input(
                "Chunk Size",
                min_value=256,
                max_value=4096,
                step=128,
                value=int(current["chunk_size"]),
            )
            llm_temperature = st.slider(
                "LLM Temperature",
                min_value=0.0,
                max_value=1.0,
                step=0.01,
                value=float(current["llm_temperature"]),
            )
            llm_top_p = st.slider(
                "LLM Top-P (nucleus)",
                min_value=0.0,
                max_value=1.0,
                step=0.01,
                value=float(current.get("llm_top_p", 0.9)),
                help="Nucleus sampling probability mass."
            )
        with col_right:
            llm_n_ctx = st.number_input(
                "Context Window",
                min_value=1024,
                max_value=32768,
                step=256,
                value=int(current["llm_n_ctx"]),
                help="n_ctx : Total context budget (input + output tokens)."
            )
            llm_max_tokens = st.number_input(
                "LLM Max Tokens",
                min_value=256,
                max_value=4096,
                step=128,
                value=int(current["llm_max_tokens"]),
                help="Generation budget for model outputs."
            )
            llm_repeat_penalty = st.number_input(
                "Repeat Penalty",
                min_value=0.5,
                max_value=2.0,
                step=0.05,
                value=float(current.get("llm_repeat_penalty", 1.1)),
                help="Discourage verbatim repetition (>1.0)."
            )
        st.subheader("Performance")
        max_workers = st.number_input(
            "Max Workers",
            min_value=1,
            max_value=16,
            step=1,
            value=int(current["max_workers"]),
        )
        llm_n_threads = st.number_input(
            "LLM CPU Threads",
            min_value=1,
            max_value=64,
            step=1,
            value=int(current.get("llm_n_threads", 4)),
            help="CPU threads used by llama.cpp when applicable."
        )
        with st.expander("GPU Controls", expanded=False):
            gpu_backend_options = ["auto", "metal", "cuda", "cpu"]
            gpu_backend = st.selectbox(
                "GPU Backend",
                options=gpu_backend_options,
                index=gpu_backend_options.index(str(current["gpu_backend"]).lower()) if str(current["gpu_backend"]).lower() in gpu_backend_options else 0,
                help="Choose the inference backend. Use 'auto' to detect automatically."
            )
            llm_n_gpu_layers = st.number_input(
                "LLM GPU Layers",
                min_value=-1,
                max_value=80,
                step=1,
                value=int(current["llm_n_gpu_layers"]),
                help="-1 loads all layers on the GPU; set a concrete number to cap GPU usage."
            )
            gpu_memory_fraction = st.slider(
                "GPU Memory Fraction",
                min_value=0.1,
                max_value=1.0,
                step=0.05,
                value=float(current["gpu_memory_fraction"]),
                help="Fraction of total GPU memory the model may reserve."
            )

        submitted = st.form_submit_button("Apply Settings")

    if submitted:
        new_values = {
            "quality_threshold": float(quality_threshold),
            "confidence_threshold": float(confidence_threshold),
            "chunk_size": int(chunk_size),
            "llm_n_ctx": int(llm_n_ctx),
            "llm_temperature": float(llm_temperature),
            "llm_top_p": float(llm_top_p),
            "llm_repeat_penalty": float(llm_repeat_penalty),
            "llm_max_tokens": int(llm_max_tokens),
            "llm_n_threads": int(llm_n_threads),
            "max_workers": int(max_workers),
            "enable_gpu": str(gpu_backend).lower() != "cpu",
            "gpu_backend": str(gpu_backend),
            "llm_n_gpu_layers": int(llm_n_gpu_layers),
            "gpu_memory_fraction": float(gpu_memory_fraction),
        }
        _apply_settings(new_values)
        st.sidebar.success("Settings updated. Next run will use the new values.")

    st.sidebar.markdown("---")
    # Heuristic guidance: ~4 chars per token for English text
    try:
        approx_input_tokens = max(1, int(int(current["chunk_size"]) / 4))
        approx_total = approx_input_tokens + int(current["llm_max_tokens"])
        n_ctx_val = int(current.get("llm_n_ctx", config.llm_n_ctx))
        if approx_total > n_ctx_val:
            st.sidebar.warning(
                f"Estimated tokens (input≈{approx_input_tokens} + output={int(current['llm_max_tokens'])} = {approx_total}) exceed n_ctx={n_ctx_val}. Consider reducing chunk size or max tokens."
            )
    except Exception:
        pass

    st.sidebar.caption(
        f"Environment: **{config.environment.value.capitalize()}** · "
        f"Backend: `{config.gpu_backend}` · "
        f"GPU Layers: {config.llm_n_gpu_layers} · "
        f"n_ctx: {int(current.get('llm_n_ctx', config.llm_n_ctx))}"
    )


def _handle_uploaded_file(uploaded_file) -> Dict[str, Any]:
    suffix = Path(uploaded_file.name).suffix or ".tmp"
    data = uploaded_file.read()
    if not data:
        raise ProcessingError("Uploaded file is empty.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(data)
        tmp_path = tmp.name

    try:
        processor = _get_processor()
        result = asyncio.run(processor.process_document(tmp_path))
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    entities_df = _process_entities(result.extraction_result.entities)
    payload = {
        "document_id": result.document_id,
        "document_type": result.document_type,
        "confidence_score": result.extraction_result.confidence_score,
        "processing_time": result.processing_time,
        "strategy_used": result.extraction_result.strategy_used,
        "metadata": result.metadata,
        "entities": entities_df,
    }
    return payload


def main() -> None:
    _ensure_session_defaults()
    _render_sidebar()

    st.title("Industrial Procedural Knowledge Extraction (IPKE)")
    st.write("Upload a document, tune extraction parameters, and review structured results.")

    config = get_config()
    supported_types = sorted({ext.lstrip(".") for ext in config.supported_formats})
    uploaded_file = st.file_uploader(
        "Upload a document",
        type=supported_types or None,
        help="Supported formats are driven by the system configuration.",
    )

    run_extraction = st.button("Run Extraction", type="primary")

    if run_extraction:
        if uploaded_file is None:
            st.warning("Please select a document before running extraction.")
        else:
            with st.spinner("Processing document..."):
                try:
                    payload = _handle_uploaded_file(uploaded_file)
                    st.session_state["last_result"] = payload
                except ProcessingError as exc:
                    st.error(f"Processing failed: {exc}")
                except RuntimeError as exc:
                    st.error(f"Model runtime error: {exc}")
                except Exception as exc:
                    st.exception(exc)

    last_result = st.session_state.get("last_result")
    if last_result:
        col_left, col_right = st.columns(2)
        col_left.metric("Entities Extracted", len(last_result["entities"]))
        col_left.metric(
            "Confidence Score",
            f"{last_result['confidence_score']:.2f}",
        )
        col_right.metric(
            "Processing Time (s)",
            f"{last_result['processing_time']:.2f}",
        )
        col_right.metric("Strategy", last_result["strategy_used"])

        st.subheader("Entities")
        st.dataframe(last_result["entities"], use_container_width=True, hide_index=True)

        with st.expander("Metadata and Diagnostics", expanded=False):
            st.json(
                {
                    "document_id": last_result["document_id"],
                    "document_type": last_result["document_type"],
                    "metadata": last_result["metadata"],
                    "settings": st.session_state["settings"],
                    "processor_stats": _get_processor().get_processing_stats(),
                }
            )


if __name__ == "__main__":
    main()
