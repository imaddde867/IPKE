"""
EXPLAINIUM - Knowledge Table Frontend

A clean, modern frontend for displaying extracted knowledge.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Any, List, Optional
import json
from datetime import datetime
import io
import sys
import os

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, os.path.join(project_root, 'src'))

# Try to import AI components
try:
    from ai.advanced_knowledge_engine import AdvancedKnowledgeEngine
    from core.config import AIConfig
    AI_AVAILABLE = True
except ImportError:
    AI_AVAILABLE = False

def process_document(uploaded_file, use_ai=False):
    """Process uploaded document and extract knowledge"""
    try:
        # Read file content
        if uploaded_file.type == "application/pdf":
            try:
                import PyPDF2
                pdf_reader = PyPDF2.PdfReader(io.BytesIO(uploaded_file.read()))
                content = ""
                for page in pdf_reader.pages:
                    content += page.extract_text() + "\n"
            except:
                content = "PDF content could not be extracted"
        else:
            content = str(uploaded_file.read(), "utf-8")
        
        # Extract knowledge from content
        knowledge_items = extract_knowledge_from_text(content, uploaded_file.name)
        
        return knowledge_items
        
    except Exception as e:
        st.error(f"Error processing document: {e}")
        return []

def extract_knowledge_from_text(text, source_name):
    """Extract knowledge items from text content"""
    import re
    
    knowledge_items = []
    words = text.lower().split()
    
    # Extract processes (look for action words and procedures)
    process_patterns = [
        r'\b\w+ing\b',  # words ending in -ing
        r'\bprocess\w*\b',
        r'\bprocedure\w*\b',
        r'\bmethod\w*\b',
        r'\bstep\w*\b'
    ]
    
    processes = set()
    for pattern in process_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        processes.update(matches[:3])  # Limit to 3
    
    for process in list(processes)[:3]:
        knowledge_items.append({
            "Knowledge": process.title(),
            "Type": "processes",
            "Confidence": round(0.75 + len(process) * 0.01, 2),
            "Category": "Process",
            "Description": f"Process identified in {source_name}",
            "Source": source_name,
            "Extracted_At": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
    
    # Extract requirements (look for obligation words)
    requirement_indicators = ['must', 'shall', 'required', 'mandatory', 'essential']
    for indicator in requirement_indicators:
        if indicator in text.lower():
            knowledge_items.append({
                "Knowledge": f"{indicator.title()} Requirement",
                "Type": "requirements",
                "Confidence": 0.85,
                "Category": "Requirement",
                "Description": f"Requirement with '{indicator}' found in document",
                "Source": source_name,
                "Extracted_At": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
    
    # Extract systems (look for capitalized terms)
    systems = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
    unique_systems = list(set(systems))[:3]
    
    for system in unique_systems:
        if len(system) > 3:  # Filter out short words
            knowledge_items.append({
                "Knowledge": system,
                "Type": "systems",
                "Confidence": 0.80,
                "Category": "System",
                "Description": f"System component identified in {source_name}",
                "Source": source_name,
                "Extracted_At": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
    
    # Extract risks (look for risk-related terms)
    risk_terms = ['risk', 'hazard', 'danger', 'threat', 'vulnerability']
    for term in risk_terms:
        if term in text.lower():
            knowledge_items.append({
                "Knowledge": f"{term.title()} Factor",
                "Type": "risks",
                "Confidence": 0.75,
                "Category": "Risk",
                "Description": f"Risk factor containing '{term}' identified",
                "Source": source_name,
                "Extracted_At": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
    
    # Extract key concepts (most frequent meaningful words)
    word_freq = {}
    for word in words:
        if len(word) > 4 and word.isalpha():
            word_freq[word] = word_freq.get(word, 0) + 1
    
    top_concepts = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:2]
    for word, freq in top_concepts:
        knowledge_items.append({
            "Knowledge": word.title(),
            "Type": "concepts",
            "Confidence": min(0.90, 0.60 + freq * 0.02),
            "Category": "Concept",
            "Description": f"Key concept mentioned {freq} times",
            "Source": source_name,
            "Extracted_At": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
    
    return knowledge_items

def get_demo_data():
    """Get demo knowledge data"""
    return [
        {
            "Knowledge": "Customer Onboarding Process",
            "Type": "processes",
            "Confidence": 0.95,
            "Category": "Business Process",
            "Description": "Multi-step workflow for bringing new customers into the system",
            "Source": "Demo Data",
            "Extracted_At": "2024-01-15 10:30:00"
        },
        {
            "Knowledge": "CRM System",
            "Type": "systems",
            "Confidence": 0.92,
            "Category": "Software System",
            "Description": "Customer relationship management platform",
            "Source": "Demo Data",
            "Extracted_At": "2024-01-15 10:32:00"
        },
        {
            "Knowledge": "Data Privacy Compliance",
            "Type": "requirements",
            "Confidence": 0.85,
            "Category": "Legal Requirement",
            "Description": "Must comply with GDPR and local data protection laws",
            "Source": "Demo Data",
            "Extracted_At": "2024-01-15 10:33:00"
        },
        {
            "Knowledge": "Technical Complexity Risk",
            "Type": "risks",
            "Confidence": 0.78,
            "Category": "Technical Risk",
            "Description": "Risk of technical requirements exceeding capabilities",
            "Source": "Demo Data",
            "Extracted_At": "2024-01-15 10:34:00"
        },
        {
            "Knowledge": "Solution Architecture",
            "Type": "concepts",
            "Confidence": 0.82,
            "Category": "Technical Concept",
            "Description": "Overall system design and component relationships",
            "Source": "Demo Data",
            "Extracted_At": "2024-01-15 10:35:00"
        }
    ]

def main():
    """Main application"""
    st.set_page_config(
        page_title="EXPLAINIUM Knowledge Table",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🧠 EXPLAINIUM Knowledge Table")
    st.markdown("**Deep Knowledge Extraction & Analysis Dashboard**")
    
    # Initialize session state
    if 'knowledge_data' not in st.session_state:
        st.session_state.knowledge_data = get_demo_data()
    
    # Sidebar controls
    with st.sidebar:
        st.header("📊 Controls")
        
        # AI Status
        if AI_AVAILABLE:
            st.success("✅ AI Engine Available")
        else:
            st.warning("⚠️ AI Engine Unavailable")
        
        # File upload
        st.subheader("📄 Upload Document")
        uploaded_file = st.file_uploader(
            "Choose a file",
            type=['pdf', 'txt', 'docx'],
            help="Upload documents for knowledge extraction"
        )
        
        if uploaded_file is not None:
            if st.button("🚀 Process Document", type="primary"):
                with st.spinner("Processing document..."):
                    new_knowledge = process_document(uploaded_file, AI_AVAILABLE)
                    if new_knowledge:
                        # Add new knowledge to existing data
                        st.session_state.knowledge_data.extend(new_knowledge)
                        st.success(f"✅ Extracted {len(new_knowledge)} knowledge items!")
                        st.rerun()
        
        # Filters
        st.subheader("🔍 Filters")
        
        knowledge_types = st.multiselect(
            "Knowledge Types",
            ["concepts", "processes", "systems", "requirements", "risks"],
            default=["concepts", "processes", "systems", "requirements", "risks"]
        )
        
        confidence_range = st.slider(
            "Confidence Range",
            min_value=0.0,
            max_value=1.0,
            value=(0.0, 1.0),
            step=0.1
        )
        
        search_term = st.text_input("🔍 Search", placeholder="Search knowledge...")
        
        # Clear data
        if st.button("🗑️ Clear All Data"):
            st.session_state.knowledge_data = get_demo_data()
            st.rerun()
    
    # Main content
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.header("📊 Knowledge Table")
        
        # Get and filter data
        df = pd.DataFrame(st.session_state.knowledge_data)
        
        # Apply filters
        if knowledge_types:
            df = df[df['Type'].isin(knowledge_types)]
        
        df = df[
            (df['Confidence'] >= confidence_range[0]) & 
            (df['Confidence'] <= confidence_range[1])
        ]
        
        if search_term:
            df = df[df['Knowledge'].str.contains(search_term, case=False, na=False)]
        
        # Display data
        if df.empty:
            st.info("No data matches your filters. Try adjusting the filters.")
        else:
            st.dataframe(
                df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Confidence": st.column_config.ProgressColumn(
                        "Confidence",
                        help="Extraction confidence score",
                        min_value=0,
                        max_value=1,
                    )
                }
            )
            
            # Export
            csv = df.to_csv(index=False)
            st.download_button(
                "📥 Download CSV",
                csv,
                f"knowledge_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                "text/csv"
            )
    
    with col2:
        st.header("📈 Analytics")
        
        if not df.empty:
            # Type distribution
            type_counts = df['Type'].value_counts()
            fig = px.pie(
                values=type_counts.values,
                names=type_counts.index,
                title="Knowledge Distribution"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Confidence distribution
            fig2 = px.histogram(
                df,
                x='Confidence',
                title="Confidence Distribution",
                nbins=10
            )
            st.plotly_chart(fig2, use_container_width=True)
        
        # Stats
        st.subheader("📊 Statistics")
        st.metric("Total Items", len(df))
        if not df.empty:
            st.metric("Avg Confidence", f"{df['Confidence'].mean():.2f}")
            st.metric("Types", df['Type'].nunique())

if __name__ == "__main__":
    main()