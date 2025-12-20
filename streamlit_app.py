"""
IPKE - Industrial Procedural Knowledge Extraction
Thesis Demonstration Interface | Turku University of Applied Sciences, 2025
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from src.ai.llm_env_setup import *  # noqa: F401,F403

import asyncio
import tempfile
import json
import base64
from pathlib import Path
from typing import Dict, Any, List, Tuple

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from dataclasses import asdict

from src.core.unified_config import get_config
from src.processors.streamlined_processor import StreamlinedDocumentProcessor
from src.utils.visualizer import generate_interactive_graph_html
from src.validation import validate_extraction
from src.validation.constraint_validator import (
    ValidationReport,
    validate_constraints as validate_semantic_constraints,
)
import networkx as nx
from src.ai.types import ExtractionResult, ExtractedEntity


# ═══════════════════════════════════════════════════════════════════════════════
# SESSION & PROCESSING
# ═══════════════════════════════════════════════════════════════════════════════

def init_session():
    """Initialize session state."""
    if "result" not in st.session_state:
        st.session_state.result = None
    if "processor" not in st.session_state:
        st.session_state.processor = None


def get_processor() -> StreamlinedDocumentProcessor:
    """Get or create processor."""
    if st.session_state.processor is None:
        st.session_state.processor = StreamlinedDocumentProcessor()
    return st.session_state.processor


def load_demo() -> Dict[str, Any]:
    """Load 3M OEM SOP demo data."""
    path = Path("datasets/archive/gold_human/3M_OEM_SOP.json")
    with open(path) as f:
        data = json.load(f)
    
    entities = []
    for cat, items in data.get("resources_catalog", {}).items():
        for item in items:
            entities.append(ExtractedEntity(
                content=item.get("canonical_name", item.get("id")),
                entity_type="Resource",
                category=cat.title(),
                confidence=1.0,
                context=f"Gold: {item.get('id')}"
            ))
    
    result = ExtractionResult(
        entities=entities,
        confidence_score=1.0,
        processing_time=0.0,
        strategy_used="P3 Two-Stage + DSC",
        steps=data.get("steps", []),
        metadata={
            "relations": data.get("relations", {}),
            "procedure_info": data.get("procedure", {}),
            "resources_catalog": data.get("resources_catalog", {})
        }
    )
    
    return {"result": result, "source": "3M_OEM_SOP (Gold Standard)"}


def process_file(uploaded_file) -> Dict[str, Any]:
    """Process uploaded file."""
    suffix = Path(uploaded_file.name).suffix or ".tmp"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name
    
    try:
        processor = get_processor()
        doc_result = asyncio.run(processor.process_document(tmp_path))
        return {"result": doc_result.extraction_result, "source": doc_result.document_id}
    finally:
        os.unlink(tmp_path)


def calculate_metrics(result: ExtractionResult) -> Dict[str, Any]:
    """Calculate IPKE metrics: Φ = w_c×C_cov + w_s×S_F1 + w_o×τ"""
    steps = result.steps or []
    metadata = result.metadata or {}
    
    # Count constraints
    constraint_count = 0
    safety_count = 0
    for step in steps:
        for ctype in ["precondition", "postcondition", "guard", "warning"]:
            constraint_count += len(step.get("constraints", {}).get(ctype, []))
        if step.get("flags", {}).get("safety_critical"):
            safety_count += 1
    
    step_count = len(steps)
    relations = metadata.get("relations", {})
    sequence_count = len(relations.get("sequence", []))
    gateway_count = len(relations.get("gateways", []))
    
    # Constraint coverage estimate (constraints per step, capped at 1.0)
    c_cov = min(1.0, constraint_count / max(1, step_count))
    # Proxy Step F1: prefer richer procedures
    s_f1 = 0.85 if step_count >= 6 else (0.75 if step_count >= 3 else 0.6)
    # Order consistency proxy (τ): ratio of forward NEXT edges
    def _order_tau() -> float:
        if step_count <= 1:
            return 1.0
        step_index = {s.get("id", f"S{i+1}"): i for i, s in enumerate(steps)}
        edges = relations.get("sequence", []) if isinstance(relations, dict) else []
        if not edges:
            return 0.5
        forward = 0
        total = 0
        for link in edges:
            u, v = link.get("from"), link.get("to")
            if not u or not v or u not in step_index or v not in step_index:
                continue
            total += 1
            if step_index[v] > step_index[u]:
                forward += 1
        return (forward / total) if total else 0.5
    tau = _order_tau()
    
    # Procedural Fidelity (thesis weights: 0.5, 0.3, 0.2)
    phi = 0.5 * c_cov + 0.3 * s_f1 + 0.2 * tau
    
    return {
        "phi": phi, "c_cov": c_cov, "s_f1": s_f1, "tau": tau,
        "steps": step_count, "constraints": constraint_count,
        "sequences": sequence_count, "gateways": gateway_count,
        "safety_critical": safety_count
    }


def _build_schema_payload(result: ExtractionResult) -> Dict[str, Any]:
    """Approximate the Tier-B schema payload from an ExtractionResult for validation."""
    steps = []
    id_map: Dict[str, str] = {}
    for i, step in enumerate(result.steps or []):
        sid = step.get("id", f"S{i+1}")
        text = step.get("label") or step.get("text") or step.get("description") or ""
        if not text:
            continue
        nsid = f"S{len(steps)+1}"
        id_map[sid] = nsid
        steps.append({"id": nsid, "text": text})

    # Constraints as "conditions" with attached refs
    conditions: List[Dict[str, Any]] = []
    raw_steps = result.steps or []
    for s in raw_steps:
        sid = s.get("id")
        if not sid:
            continue
        constraints = s.get("constraints", {}) if isinstance(s.get("constraints"), dict) else {}
        for ctype, items in constraints.items():
            if not isinstance(items, list):
                continue
            for idx, item in enumerate(items):
                text = item if isinstance(item, str) else (item.get("text") or item.get("expression") or str(item))
                conditions.append({
                    "id": f"C{len(conditions)+1}",
                    "type": ctype.upper(),
                    "expression": text,
                    "attached_to": id_map.get(sid, sid),
                })

    # NEXT edges
    edges: List[Dict[str, Any]] = []
    rel = result.metadata.get("relations", {}) if isinstance(result.metadata, dict) else {}
    for link in rel.get("sequence", []) if isinstance(rel, dict) else []:
        u, v = link.get("from"), link.get("to")
        if not u or not v:
            continue
        edges.append({"from_id": id_map.get(u, u), "to_id": id_map.get(v, v), "type": "NEXT"})
    for cond in conditions:
        edges.append({"from_id": cond["id"], "to_id": cond.get("attached_to", ""), "type": "CONDITION_ON"})

    doc_id = (result.metadata or {}).get("document_id") or "unknown_document"
    return {
        "document_id": str(doc_id),
        "document_type": (result.metadata or {}).get("document_type") or "unknown",
        "title": None,
        "steps": steps,
        "conditions": conditions,
        "equipment": [],
        "parameters": [],
        "edges": edges,
        "metadata": {"source": "streamlit_ui"},
    }


def compute_graph_stats(result: ExtractionResult) -> Dict[str, Any]:
    """Compute structural completeness statistics for the PKG graph."""
    steps = result.steps or []
    rel = result.metadata.get("relations", {}) if isinstance(result.metadata, dict) else {}
    G = nx.DiGraph()
    step_ids = [s.get("id", f"S{i+1}") for i, s in enumerate(steps)]
    G.add_nodes_from(step_ids)
    for link in rel.get("sequence", []) if isinstance(rel, dict) else []:
        u, v = link.get("from"), link.get("to")
        if u in G and v in G:
            G.add_edge(u, v)
    # Fallback to linear chain if no edges
    if G.number_of_edges() == 0 and len(step_ids) > 1:
        for i in range(len(step_ids) - 1):
            G.add_edge(step_ids[i], step_ids[i + 1])

    # Connectivity and coverage
    try:
        components = list(nx.weakly_connected_components(G))
    except Exception:
        components = [set(step_ids)] if step_ids else []
    num_components = len(components)
    largest = max((len(c) for c in components), default=0)
    coverage_main = (largest / max(1, len(step_ids))) if step_ids else 0.0

    # Acyclicity and cycles
    try:
        cycles = list(nx.simple_cycles(G))
        has_cycles = len(cycles) > 0
    except Exception:
        has_cycles = False

    # Degree-based completeness: nodes with both in and out degree
    both_deg = sum(1 for n in G.nodes if G.in_degree(n) > 0 and G.out_degree(n) > 0)
    interior_ratio = both_deg / max(1, G.number_of_nodes())

    # Edge orientation consistency (forward edges vs index order)
    index = {sid: i for i, sid in enumerate(step_ids)}
    forward = sum(1 for u, v in G.edges if index.get(v, 0) > index.get(u, 0))
    tau_like = (forward / max(1, G.number_of_edges())) if G.number_of_edges() else 1.0

    return {
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
        "components": num_components,
        "main_component_ratio": coverage_main,
        "interior_ratio": interior_ratio,
        "has_cycles": has_cycles,
        "tau_like": tau_like,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR - THESIS PARAMETERS
# ═══════════════════════════════════════════════════════════════════════════════

def render_sidebar():
    """Render sidebar with thesis-aligned parameters."""
    config = get_config()
    
    st.sidebar.header("IPKE Parameters")
    
    # Prompting Strategy - outside form for instant update
    st.sidebar.subheader("Prompting Strategy")
    strategy_display = {
        "Zero-Shot": "P0",
        "Chain-of-Thought": "P1",
        "Few-Shot": "P2",
        "Two-Stage": "P3"
    }
    strategy_names = list(strategy_display.keys())
    # Map current config to display name
    current_name = next((k for k, v in strategy_display.items() if v == config.prompting_strategy), "Zero-Shot")
    strategy_name = st.sidebar.selectbox(
        "Strategy",
        options=strategy_names,
        index=strategy_names.index(current_name),
        help="Zero-Shot: Direct extraction | CoT: Step-by-step reasoning | Few-Shot: Example-guided | Two-Stage: Decomposed extraction",
        key="strategy_select"
    )
    strategy = strategy_display[strategy_name]  # Map back to P0-P3 for config
    
    # Chunking Method - outside form for instant update
    st.sidebar.subheader("Chunking")
    methods = ["fixed", "breakpoint_semantic", "dsc"]
    current_method = config.chunking_method
    if current_method in {"dual_semantic", "parent_only"}:
        current_method = "dsc"
    if current_method not in methods:
        current_method = "fixed"
    chunking = st.sidebar.selectbox(
        "Method",
        options=methods,
        index=methods.index(current_method),
        help="fixed: Character-based | semantic: Embedding coherence | dsc: Dual Semantic",
        key="chunking_select"
    )
    
    chunk_size = config.chunk_max_chars
    dsc_k = config.dsc_threshold_k
    dsc_window = config.dsc_delta_window
    dsc_headings = config.dsc_use_headings
    sem_lambda = config.sem_lambda
    sem_window = config.sem_window_w

    # Show parameters based on chunking method
    if chunking == "fixed":
        chunk_size = st.sidebar.number_input(
            "Chunk Size", min_value=500, max_value=4000, step=100,
            value=config.chunk_max_chars, help="Characters per chunk",
            key="chunk_size_input"
        )
    elif chunking == "dsc":
        st.sidebar.caption("DSC Parameters")
        dsc_k = st.sidebar.slider("Threshold k", 0.5, 2.0, float(config.dsc_threshold_k), 0.1,
                                  help="Boundary sensitivity (higher = fewer splits)",
                                  key="dsc_k_slider")
        dsc_window = st.sidebar.number_input("Delta Window", 5, 50, config.dsc_delta_window,
                                             help="Smoothing window size",
                                             key="dsc_window_input")
        dsc_headings = st.sidebar.checkbox("Use Headings", config.dsc_use_headings,
                                           help="Respect document structure",
                                           key="dsc_headings_check")
    else:  # breakpoint_semantic
        st.sidebar.caption("Semantic Parameters")
        sem_lambda = st.sidebar.slider("Lambda (λ)", 0.0, 0.5, float(config.sem_lambda), 0.05,
                                       help="Regularization strength",
                                       key="sem_lambda_slider")
        sem_window = st.sidebar.number_input("Window Size", 5, 50, config.sem_window_w,
                                             help="Semantic window size",
                                             key="sem_window_input")
    
    # LLM Parameters
    st.sidebar.subheader("LLM")
    temperature = st.sidebar.slider("Temperature", 0.0, 1.0, float(config.llm_temperature), 0.05,
                                    help="Lower = more deterministic",
                                    key="temp_slider")
    max_tokens = st.sidebar.number_input("Max Tokens", 512, 4096, config.llm_max_tokens, 128,
                                         key="max_tokens_input")
    n_ctx = st.sidebar.number_input("Context Window", 2048, 16384, config.llm_n_ctx, 512,
                                    key="n_ctx_input")
    
    # Hardware
    st.sidebar.subheader("Hardware")
    backends = ["auto", "metal", "cuda", "cpu"]
    current_backend = config.gpu_backend.lower() if config.gpu_backend.lower() in backends else "auto"
    gpu = st.sidebar.selectbox("Backend", backends, index=backends.index(current_backend),
                               key="gpu_select")
    
    # Apply button
    if st.sidebar.button("Apply", use_container_width=True, type="primary"):
        canonical_chunking = "dual_semantic" if chunking == "dsc" else chunking
        updates = {
            "prompting_strategy": strategy,
            "chunking_method": canonical_chunking,
            "chunk_max_chars": chunk_size,
            "chunk_size": chunk_size,
            "llm_temperature": temperature,
            "llm_max_tokens": max_tokens,
            "llm_n_ctx": n_ctx,
            "gpu_backend": gpu,
        }
        if chunking == "dsc":
            updates.update({
                "dsc_threshold_k": dsc_k,
                "dsc_delta_window": dsc_window,
                "dsc_use_headings": dsc_headings,
            })
        elif chunking == "breakpoint_semantic":
            updates.update({
                "sem_lambda": sem_lambda,
                "sem_window_w": sem_window,
            })
        for key, value in updates.items():
            setattr(config, key, value)
        st.session_state.processor = None  # Reset processor
        st.sidebar.success("Parameters updated")
    
    # Current config display
    st.sidebar.divider()
    strategy_labels = {"P0": "Zero-Shot", "P1": "CoT", "P2": "Few-Shot", "P3": "Two-Stage"}
    chunking_label = "dsc" if config.chunking_method in {"dual_semantic", "parent_only"} else config.chunking_method
    st.sidebar.caption(f"**Model:** Mistral-7B (Q4_K_M)")
    st.sidebar.caption(f"**Strategy:** {strategy_labels.get(config.prompting_strategy, config.prompting_strategy)} | **Chunking:** {chunking_label}")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN UI
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    st.set_page_config(
        page_title="IPKE - Industrial PKG Extraction",
        page_icon="🔬",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    init_session()
    render_sidebar()
    
    # Header
    st.markdown("""
    <div style="margin-bottom: 24px;">
        <h1 style="margin: 0; font-size: 2.2rem; font-weight: 700;">
            IPKE <span style="font-weight: 400; color: #64748B; font-size: 1.2rem;">Industrial Procedural Knowledge Extraction</span>
        </h1>
        <p style="color: #94A3B8; margin-top: 8px; font-size: 1rem;">
            Transforming Standard Operating Procedures into machine-actionable, AI-queryable knowledge graphs
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Thesis metrics summary
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Constraint Coverage", "75%", help="vs 50% baseline (70B model)")
    col2.metric("Procedural Fidelity", "0.70", help="Φ composite metric")
    col3.metric("Model Size", "7B", help="Mistral-7B-Instruct")
    col4.metric("Efficiency", "10×", help="vs 70B model")
    
    st.divider()
    
    # Input Section
    config = get_config()
    supported = sorted({ext.lstrip(".") for ext in config.supported_formats})
    
    col1, col2 = st.columns([3, 1])
    with col1:
        uploaded = st.file_uploader("Upload Document", type=supported or None)
    with col2:
        # Add vertical spacing to align with file uploader
        st.markdown("<div style='height: 38px'></div>", unsafe_allow_html=True)
        demo_btn = st.button("Load Demo", use_container_width=True)
        if uploaded:
            extract_btn = st.button("Extract", type="primary", use_container_width=True)
        else:
            extract_btn = False
    
    # Process
    if demo_btn:
        with st.spinner("Loading demo..."):
            st.session_state.result = load_demo()
            st.success("Demo loaded")
    
    if extract_btn and uploaded:
        with st.spinner("Running IPKE pipeline..."):
            try:
                st.session_state.result = process_file(uploaded)
                st.success("Extraction complete")
            except Exception as e:
                st.error(f"Failed: {e}")
    
    # Results
    if st.session_state.result:
        result = st.session_state.result["result"]
        source = st.session_state.result["source"]
        
        st.divider()
        st.subheader(f"Results: {source}")
        
        # Metrics
        metrics = calculate_metrics(result)
        
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Φ (Fidelity)", f"{metrics['phi']:.2f}")
        col2.metric("Steps", metrics["steps"])
        col3.metric("Constraints", metrics["constraints"])
        col4.metric("Sequences", metrics["sequences"])
        col5.metric("Safety-Critical", metrics["safety_critical"])
        
        st.divider()
        
        # Tabs
        tab1, tab2, tabQ, tab3 = st.tabs(["Graph", "Data", "Quality", "Debug"])
        
        with tab1:
            st.subheader("Procedural Knowledge Graph")
            steps = result.steps or []
            constraints_count = metrics["constraints"]
            safety_count = metrics["safety_critical"]
            
            # Info bar with graph statistics
            col_a, col_b, col_c, col_d = st.columns(4)
            col_a.metric("Steps", len(steps))
            col_b.metric("Constraints", constraints_count)
            col_c.metric("Safety-Critical", safety_count)
            col_d.metric("Edges", metrics["sequences"])
            
            st.markdown("---")
            
            try:
                html = generate_interactive_graph_html(result, height="100vh")
                
                # Embed the graph directly in the page
                st.markdown("""
                <style>
                    .stIFrame { border-radius: 12px; overflow: hidden; }
                    iframe[title="streamlit_app.components.v1.html"] { 
                        border: 1px solid rgba(148, 163, 184, 0.15); 
                        border-radius: 12px;
                    }
                </style>
                """, unsafe_allow_html=True)
                
                # Embedded visualization (primary)
                components.html(html, height=700, scrolling=False)
                
                # Option to open fullscreen
                b64 = base64.b64encode(html.encode()).decode()
                fullscreen_js = f"""
                <script>
                function openFullscreen() {{
                    var html = atob("{b64}");
                    var blob = new Blob([html], {{type: 'text/html'}});
                    window.open(URL.createObjectURL(blob), '_blank');
                }}
                </script>
                <div style="display: flex; gap: 12px; margin-top: 12px;">
                    <button onclick="openFullscreen()" style="
                        padding: 10px 20px;
                        background: linear-gradient(135deg, #2563EB 0%, #1D4ED8 100%);
                        color: #fff;
                        border: none;
                        border-radius: 8px;
                        cursor: pointer;
                        font-size: 14px;
                        font-weight: 500;
                        display: flex;
                        align-items: center;
                        gap: 8px;
                        transition: all 0.2s ease;
                        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.3);
                    ">
                        Open Fullscreen PKG
                    </button>
                </div>
                """
                components.html(fullscreen_js, height=60)
            except Exception as e:
                st.error(f"Graph visualization failed: {e}")
                import traceback
                st.code(traceback.format_exc())
        
        with tab2:
            st.subheader("Extracted Data")
            
            # Resources/Entities
            if result.entities:
                st.markdown("#### Resources & Entities")
                entities_df = pd.DataFrame([
                    {"Resource": e.content, "Category": e.category, "Confidence": f"{e.confidence:.0%}"}
                    for e in result.entities
                ])
                st.dataframe(entities_df, use_container_width=True, hide_index=True)
            
            # Steps with detailed view
            if result.steps:
                st.markdown("#### Procedural Steps")
                for idx, step in enumerate(result.steps):
                    step_id = step.get('id', f'S{idx+1}')
                    label = step.get('label', '')
                    action = step.get('action_verb', 'N/A').upper()
                    is_safety = step.get('flags', {}).get('safety_critical', False)
                    
                    # Get resources
                    resources = step.get('resources', {})
                    tools = resources.get('tools', [])
                    materials = resources.get('materials', [])
                    
                    # Get parameters
                    params = step.get('parameters', [])
                    
                    # Get constraints count
                    constraints = step.get('constraints', {})
                    constraint_count = sum(len(v) for v in constraints.values() if isinstance(v, list))
                    
                    # Header with safety badge
                    safety_badge = "[SAFETY] " if is_safety else ""
                    with st.expander(f"**{safety_badge}{step_id}**: {action} — {label[:60]}..."):
                        col1, col2 = st.columns([2, 1])
                        
                        with col1:
                            st.markdown(f"**Full Description:** {label}")
                            
                            obj = step.get('action_object')
                            if obj:
                                obj_name = obj.get('canonical', obj) if isinstance(obj, dict) else obj
                                st.markdown(f"**Target Object:** {obj_name}")
                        
                        with col2:
                            if is_safety:
                                st.warning("Safety-Critical Step")
                            
                            st.markdown(f"**Constraints:** {constraint_count}")
                            st.markdown(f"**Resources:** {len(tools) + len(materials)}")
                        
                        # Resources
                        if tools or materials:
                            st.markdown("---")
                            st.markdown("**Resources:**")
                            if tools:
                                st.markdown(f"- Tools: {', '.join(tools)}")
                            if materials:
                                st.markdown(f"- Materials: {', '.join(materials)}")
                        
                        # Parameters
                        if params:
                            st.markdown("---")
                            st.markdown("**Parameters:**")
                            for p in params:
                                if isinstance(p, dict):
                                    st.markdown(f"- {p.get('name', '?')}: **{p.get('value', '?')}** {p.get('unit', '')}")
            
            # Metadata section
            if result.metadata:
                st.markdown("#### Graph Metadata")
                relations = result.metadata.get('relations', {})
                if relations:
                    seq_count = len(relations.get('sequence', []))
                    gw_count = len(relations.get('gateways', []))
                    st.markdown(f"- **Sequence Edges:** {seq_count}")
                    st.markdown(f"- **Decision Gateways:** {gw_count}")
        
        with tabQ:
            st.subheader("Quality & Completeness")
            # Procedural Fidelity breakdown
            with st.container():
                st.markdown("#### Procedural Fidelity Φ")
                st.caption("Φ = 0.5·ConstraintCoverage + 0.3·StepF1 + 0.2·Order Consistency")
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Φ", f"{metrics['phi']:.3f}")
                c2.metric("ConstraintCoverage", f"{metrics['c_cov']:.3f}")
                c3.metric("StepF1 (proxy)", f"{metrics['s_f1']:.3f}")
                c4.metric("Order Consistency", f"{metrics['tau']:.3f}")

            st.markdown("---")
            # Schema validation
            st.markdown("#### Schema Validation")
            try:
                payload = _build_schema_payload(result)
                is_valid, issues = validate_extraction(payload, autofix=True)
                auto_fixes = [msg for msg in issues if str(msg).startswith("AUTO-FIX:")]
                schema_errors = [msg for msg in issues if not str(msg).startswith("AUTO-FIX:")]
                c1, c2, c3 = st.columns(3)
                c1.metric("Valid", "Yes" if is_valid else "No")
                c2.metric("Auto-fixes", len(auto_fixes))
                c3.metric("Errors", len(schema_errors))
                if schema_errors:
                    st.warning("Schema issues detected")
                    st.code("\n".join(schema_errors[:10]) + ("\n…" if len(schema_errors) > 10 else ""))
                elif auto_fixes:
                    st.info("Auto-fixes applied")
                    st.code("\n".join(auto_fixes[:10]) + ("\n…" if len(auto_fixes) > 10 else ""))
            except Exception as e:
                st.error(f"Validation failed: {e}")

            st.markdown("---")
            # Semantic constraint validation
            st.markdown("#### Constraint Validation (Semantic)")
            try:
                # Gather constraints from steps if top-level is empty
                constraints: List[Dict[str, object]] = []
                if result.constraints:
                    constraints.extend(result.constraints)
                for s in (result.steps or []):
                    cdict = s.get("constraints", {}) if isinstance(s.get("constraints"), dict) else {}
                    for ctype, items in cdict.items():
                        if not isinstance(items, list):
                            continue
                        for item in items:
                            text = item if isinstance(item, str) else (item.get("text") or item.get("expression") or str(item))
                            constraints.append({
                                "id": item.get("id") if isinstance(item, dict) else None,
                                "type": ctype,
                                "text": text,
                                "attached_to": s.get("id"),
                            })
                report: ValidationReport = validate_semantic_constraints(constraints)
                c1, c2, c3 = st.columns(3)
                c1.metric("Passed", len(report.passed))
                c2.metric("Warnings", len(report.warnings))
                c3.metric("Errors", len(report.errors))
                if report.errors:
                    st.error("Top Errors")
                    st.code("\n".join(f"{cid}: {msg}" for cid, msg in report.errors[:10]))
                if report.warnings:
                    st.warning("Top Warnings")
                    st.code("\n".join(f"{cid}: {msg}" for cid, msg in report.warnings[:10]))
            except Exception as e:
                st.error(f"Constraint checks failed: {e}")

            st.markdown("---")
            # Graph completeness metrics
            st.markdown("#### Graph Completeness")
            try:
                gstats = compute_graph_stats(result)
                d1, d2, d3, d4 = st.columns(4)
                d1.metric("Nodes", gstats["nodes"])
                d2.metric("Edges", gstats["edges"])
                d3.metric("Components", gstats["components"])
                d4.metric("Main Component", f"{gstats['main_component_ratio']:.0%}")
                e1, e2, e3 = st.columns(3)
                e1.metric("Interior Nodes", f"{gstats['interior_ratio']:.0%}")
                e2.metric("Forward Edges", f"{gstats['tau_like']:.0%}")
                e3.metric("Cycles", "Yes" if gstats["has_cycles"] else "No")
            except Exception as e:
                st.error(f"Graph analysis failed: {e}")

            st.markdown("---")
            # Exports
            st.markdown("#### Export")
            try:
                raw_json = json.dumps(asdict(result), ensure_ascii=False, indent=2)
            except Exception:
                # Fallback manual serialization
                raw_json = json.dumps({
                    "entities": [e.__dict__ for e in (result.entities or [])],
                    "steps": result.steps or [],
                    "constraints": result.constraints or [],
                    "confidence_score": result.confidence_score,
                    "processing_time": result.processing_time,
                    "strategy_used": result.strategy_used,
                    "quality_metrics": result.quality_metrics,
                    "metadata": result.metadata,
                }, ensure_ascii=False, indent=2)
            payload_json = json.dumps(_build_schema_payload(result), ensure_ascii=False, indent=2)
            colx, coly = st.columns(2)
            colx.download_button("Download Extraction JSON", raw_json, file_name=f"{source}_extraction.json", mime="application/json")
            coly.download_button("Download Schema Payload", payload_json, file_name=f"{source}_schema.json", mime="application/json")

        with tab3:
            st.subheader("Debug")
            config = get_config()
            st.json({
                "source": source,
                "strategy": config.prompting_strategy,
                "chunking": config.chunking_method,
                "steps": len(result.steps or []),
                "constraints": metrics["constraints"],
                "phi": metrics["phi"],
                "settings": {
                    "chunk_size": config.chunk_max_chars,
                    "temperature": config.llm_temperature,
                    "max_tokens": config.llm_max_tokens,
                    "context": config.llm_n_ctx,
                    "dsc_k": config.dsc_threshold_k,
                }
            })
    
    else:
        # Empty state with better UI
        st.markdown("""
        <div style="
            text-align: center; 
            padding: 60px 40px; 
            background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%);
            border-radius: 16px;
            border: 1px solid rgba(148, 163, 184, 0.1);
            margin-top: 20px;
        ">
            <h3 style="color: #F8FAFC; font-weight: 600; margin-bottom: 8px;">Ready to Extract Knowledge</h3>
            <p style="color: #94A3B8; font-size: 1rem;">
                Upload a document or load demo data to generate an interactive<br>
                Procedural Knowledge Graph (PKG) with full AI queryability.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    # Footer
    st.divider()
    st.markdown("""
    <div style="display: flex; justify-content: space-between; align-items: center; color: #64748B; font-size: 12px;">
        <div>
            <strong style="color: #3B82F6;">IPKE</strong> — Industrial Procedural Knowledge Extraction
        </div>
        <div>
            Turku University of Applied Sciences · 2025 · 
            <a href="https://github.com/imaddde867" style="color: #3B82F6; text-decoration: none;">GitHub</a>
        </div>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
