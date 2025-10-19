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
    page_title="Explainium",
    page_icon="🐴",
    layout="wide",
)


def _ensure_session_defaults() -> None:
    config = get_config()
    defaults = {
        "quality_threshold": config.quality_threshold,
        "confidence_threshold": config.confidence_threshold,
        "chunk_size": config.chunk_size,
        "llm_temperature": config.llm_temperature,
        "llm_max_tokens": config.llm_max_tokens,
        "max_workers": config.max_workers,
        "enable_gpu": config.enable_gpu,
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
        quality_threshold = st.slider(
            "Quality Threshold",
            min_value=0.0,
            max_value=1.0,
            step=0.01,
            value=float(current["quality_threshold"]),
        )
        confidence_threshold = st.slider(
            "Confidence Threshold",
            min_value=0.0,
            max_value=1.0,
            step=0.01,
            value=float(current["confidence_threshold"]),
        )
        chunk_size = st.number_input(
            "Chunk Size",
            min_value=256,
            max_value=4096,
            step=128,
            value=int(current["chunk_size"]),
        )
        llm_temperature = st.number_input(
            "LLM Temperature",
            min_value=0.0,
            max_value=1.0,
            step=0.05,
            value=float(current["llm_temperature"]),
        )
        llm_max_tokens = st.number_input(
            "LLM Max Tokens",
            min_value=256,
            max_value=4096,
            step=128,
            value=int(current["llm_max_tokens"]),
        )
        max_workers = st.number_input(
            "Max Workers",
            min_value=1,
            max_value=16,
            step=1,
            value=int(current["max_workers"]),
        )
        enable_gpu = st.checkbox(
            "Enable GPU",
            value=bool(current["enable_gpu"]),
            help="Disable if running on CPU-only hardware.",
        )

        submitted = st.form_submit_button("Apply Settings")

    if submitted:
        new_values = {
            "quality_threshold": float(quality_threshold),
            "confidence_threshold": float(confidence_threshold),
            "chunk_size": int(chunk_size),
            "llm_temperature": float(llm_temperature),
            "llm_max_tokens": int(llm_max_tokens),
            "max_workers": int(max_workers),
            "enable_gpu": bool(enable_gpu),
        }
        _apply_settings(new_values)
        st.sidebar.success("Settings updated. Next run will use the new values.")

    st.sidebar.markdown("---")
    st.sidebar.caption(
        f"Environment: **{config.environment.value.capitalize()}** · "
        f"Model: `{config.llm_model_path}`"
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

    st.title("Explainium - Information Extraction")
    st.write("Upload a document, tune extraction parameters, and review structured results.")

    config = get_config()
    supported_types = sorted({ext.lstrip(".") for ext in config.supported_formats})
    uploaded_file = st.file_uploader(
        "Upload a document",
        type=supported_types or None,
        help="Supported formats are driven by the Explainium configuration.",
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
