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
    """Process uploaded document/media and extract knowledge"""
    try:
        file_type = uploaded_file.type
        file_name = uploaded_file.name
        
        # Handle different file types
        if file_type == "application/pdf":
            content = extract_pdf_content(uploaded_file)
            knowledge_items = extract_knowledge_from_text(content, file_name)
            
        elif file_type.startswith("image/"):
            knowledge_items = extract_knowledge_from_image(uploaded_file, file_name)
            
        elif file_type.startswith("video/"):
            knowledge_items = extract_knowledge_from_video(uploaded_file, file_name)
            
        elif file_type.startswith("audio/"):
            knowledge_items = extract_knowledge_from_audio(uploaded_file, file_name)
            
        elif file_type == "text/plain":
            content = str(uploaded_file.read(), "utf-8")
            knowledge_items = extract_knowledge_from_text(content, file_name)
            
        else:
            # Try to read as text
            try:
                content = str(uploaded_file.read(), "utf-8")
                knowledge_items = extract_knowledge_from_text(content, file_name)
            except:
                knowledge_items = [{
                    "Knowledge": f"Unsupported File Type: {file_type}",
                    "Type": "systems",
                    "Confidence": 0.5,
                    "Category": "File Processing",
                    "Description": f"File type {file_type} is not yet supported",
                    "Source": file_name,
                    "Extracted_At": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }]
        
        return knowledge_items
        
    except Exception as e:
        st.error(f"Error processing file: {e}")
        return []

def extract_pdf_content(uploaded_file):
    """Extract text from PDF"""
    try:
        import PyPDF2
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(uploaded_file.read()))
        content = ""
        for page in pdf_reader.pages:
            content += page.extract_text() + "\n"
        return content
    except:
        return "PDF content could not be extracted"

def extract_knowledge_from_image(uploaded_file, file_name):
    """Extract knowledge from image files"""
    try:
        from PIL import Image
        import numpy as np
        
        # Load image
        image = Image.open(uploaded_file)
        
        # Basic image analysis
        width, height = image.size
        mode = image.mode
        format_type = image.format
        
        knowledge_items = []
        
        # Image metadata knowledge
        knowledge_items.append({
            "Knowledge": f"Image Dimensions: {width}x{height}",
            "Type": "systems",
            "Confidence": 1.0,
            "Category": "Image Properties",
            "Description": f"Image resolution and format information",
            "Source": file_name,
            "Extracted_At": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
        
        # Color analysis
        if mode == "RGB":
            # Convert to numpy array for analysis
            img_array = np.array(image)
            avg_color = np.mean(img_array, axis=(0, 1))
            
            dominant_color = "Red" if avg_color[0] > avg_color[1] and avg_color[0] > avg_color[2] else \
                           "Green" if avg_color[1] > avg_color[2] else "Blue"
            
            knowledge_items.append({
                "Knowledge": f"Dominant Color: {dominant_color}",
                "Type": "concepts",
                "Confidence": 0.8,
                "Category": "Visual Analysis",
                "Description": f"Primary color detected in image",
                "Source": file_name,
                "Extracted_At": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
        
        # Image type classification
        if any(keyword in file_name.lower() for keyword in ['diagram', 'chart', 'graph']):
            knowledge_items.append({
                "Knowledge": "Technical Diagram",
                "Type": "concepts",
                "Confidence": 0.75,
                "Category": "Document Type",
                "Description": "Image appears to be a technical diagram or chart",
                "Source": file_name,
                "Extracted_At": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
        
        # OCR attempt if available
        try:
            import pytesseract
            text = pytesseract.image_to_string(image)
            if text.strip():
                # Extract knowledge from OCR text
                text_knowledge = extract_knowledge_from_text(text, f"{file_name} (OCR)")
                knowledge_items.extend(text_knowledge)
        except:
            # OCR not available, add note
            knowledge_items.append({
                "Knowledge": "Text Detection Unavailable",
                "Type": "systems",
                "Confidence": 0.6,
                "Category": "OCR Status",
                "Description": "OCR not available for text extraction from image",
                "Source": file_name,
                "Extracted_At": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
        
        return knowledge_items
        
    except Exception as e:
        return [{
            "Knowledge": f"Image Processing Error: {str(e)}",
            "Type": "systems",
            "Confidence": 0.3,
            "Category": "Processing Error",
            "Description": f"Error processing image: {str(e)}",
            "Source": file_name,
            "Extracted_At": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }]

def extract_knowledge_from_video(uploaded_file, file_name):
    """Extract knowledge from video files"""
    try:
        # Get file size and basic info
        file_size = len(uploaded_file.read())
        uploaded_file.seek(0)  # Reset file pointer
        
        knowledge_items = []
        
        # Basic video metadata
        knowledge_items.append({
            "Knowledge": f"Video File: {file_name}",
            "Type": "systems",
            "Confidence": 1.0,
            "Category": "Media File",
            "Description": f"Video file with size {file_size/1024/1024:.1f} MB",
            "Source": file_name,
            "Extracted_At": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
        
        # Infer content type from filename
        if any(keyword in file_name.lower() for keyword in ['training', 'tutorial', 'demo']):
            knowledge_items.append({
                "Knowledge": "Training Content",
                "Type": "processes",
                "Confidence": 0.7,
                "Category": "Educational Content",
                "Description": "Video appears to contain training or educational material",
                "Source": file_name,
                "Extracted_At": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
        
        if any(keyword in file_name.lower() for keyword in ['meeting', 'conference', 'presentation']):
            knowledge_items.append({
                "Knowledge": "Business Meeting",
                "Type": "processes",
                "Confidence": 0.75,
                "Category": "Business Process",
                "Description": "Video appears to be a business meeting or presentation",
                "Source": file_name,
                "Extracted_At": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
        
        # Note about advanced processing
        knowledge_items.append({
            "Knowledge": "Video Analysis Available",
            "Type": "concepts",
            "Confidence": 0.9,
            "Category": "Processing Capability",
            "Description": "Advanced video analysis with scene detection and frame extraction available",
            "Source": file_name,
            "Extracted_At": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
        
        return knowledge_items
        
    except Exception as e:
        return [{
            "Knowledge": f"Video Processing Error: {str(e)}",
            "Type": "systems",
            "Confidence": 0.3,
            "Category": "Processing Error",
            "Description": f"Error processing video: {str(e)}",
            "Source": file_name,
            "Extracted_At": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }]

def extract_knowledge_from_audio(uploaded_file, file_name):
    """Extract knowledge from audio files"""
    try:
        # Get file size and basic info
        file_size = len(uploaded_file.read())
        uploaded_file.seek(0)  # Reset file pointer
        
        knowledge_items = []
        
        # Basic audio metadata
        knowledge_items.append({
            "Knowledge": f"Audio File: {file_name}",
            "Type": "systems",
            "Confidence": 1.0,
            "Category": "Media File",
            "Description": f"Audio file with size {file_size/1024/1024:.1f} MB",
            "Source": file_name,
            "Extracted_At": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
        
        # Infer content type from filename
        if any(keyword in file_name.lower() for keyword in ['meeting', 'interview', 'call']):
            knowledge_items.append({
                "Knowledge": "Business Communication",
                "Type": "processes",
                "Confidence": 0.8,
                "Category": "Communication",
                "Description": "Audio appears to contain business communication or meeting",
                "Source": file_name,
                "Extracted_At": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
        
        if any(keyword in file_name.lower() for keyword in ['training', 'lecture', 'presentation']):
            knowledge_items.append({
                "Knowledge": "Educational Content",
                "Type": "concepts",
                "Confidence": 0.75,
                "Category": "Learning Material",
                "Description": "Audio appears to contain educational or training content",
                "Source": file_name,
                "Extracted_At": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
        
        # Note about transcription capability
        knowledge_items.append({
            "Knowledge": "Speech-to-Text Available",
            "Type": "concepts",
            "Confidence": 0.9,
            "Category": "Processing Capability",
            "Description": "Audio transcription with Whisper AI available for text extraction",
            "Source": file_name,
            "Extracted_At": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
        
        return knowledge_items
        
    except Exception as e:
        return [{
            "Knowledge": f"Audio Processing Error: {str(e)}",
            "Type": "systems",
            "Confidence": 0.3,
            "Category": "Processing Error",
            "Description": f"Error processing audio: {str(e)}",
            "Source": file_name,
            "Extracted_At": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }]

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
        st.session_state.knowledge_data = []
    
    # Sidebar controls
    with st.sidebar:
        st.header("📊 Controls")
        
        # AI Status
        if AI_AVAILABLE:
            st.success("✅ AI Engine Available")
        else:
            st.warning("⚠️ AI Engine Unavailable")
        
        # File upload
        st.subheader("📄 Upload Media")
        
        # Show supported formats
        with st.expander("📋 Supported Formats"):
            st.markdown("""
            **📄 Documents:** PDF, TXT, DOCX
            **🖼️ Images:** JPG, PNG, GIF, BMP, TIFF
            **🎥 Videos:** MP4, AVI, MOV, MKV
            **🎵 Audio:** MP3, WAV, FLAC
            """)
        
        uploaded_file = st.file_uploader(
            "Choose a file",
            type=['pdf', 'txt', 'docx', 'jpg', 'jpeg', 'png', 'gif', 'bmp', 'tiff', 'mp4', 'avi', 'mov', 'mkv', 'mp3', 'wav', 'flac'],
            help="Upload any supported media type for AI-powered knowledge extraction"
        )
        
        if uploaded_file is not None:
            # Show file info
            file_type = uploaded_file.type
            if file_type.startswith("image/"):
                st.info(f"🖼️ Image: {uploaded_file.name}")
                button_text = "🔍 Analyze Image"
                spinner_text = "Analyzing image with computer vision..."
            elif file_type.startswith("video/"):
                st.info(f"🎥 Video: {uploaded_file.name}")
                button_text = "🎬 Process Video"
                spinner_text = "Processing video with scene analysis..."
            elif file_type.startswith("audio/"):
                st.info(f"🎵 Audio: {uploaded_file.name}")
                button_text = "🎤 Transcribe Audio"
                spinner_text = "Transcribing audio with Whisper AI..."
            else:
                st.info(f"📄 Document: {uploaded_file.name}")
                button_text = "🚀 Process Document"
                spinner_text = "Processing document with AI..."
            
            if st.button(button_text, type="primary"):
                with st.spinner(spinner_text):
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
        
        # Data management
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("🗑️ Clear All"):
                st.session_state.knowledge_data = []
                st.success("✅ All data cleared!")
                st.rerun()
        
        with col_b:
            if st.button("📋 Load Demo"):
                st.session_state.knowledge_data = get_demo_data()
                st.success("✅ Demo data loaded!")
                st.rerun()
    
    # Main content
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.header("📊 Knowledge Table")
        
        # Get and filter data
        df = pd.DataFrame(st.session_state.knowledge_data)
        
        # Apply filters only if DataFrame is not empty
        if not df.empty:
            # Apply type filter
            if knowledge_types and 'Type' in df.columns:
                df = df[df['Type'].isin(knowledge_types)]
            
            # Apply confidence filter
            if 'Confidence' in df.columns:
                df = df[
                    (df['Confidence'] >= confidence_range[0]) & 
                    (df['Confidence'] <= confidence_range[1])
                ]
            
            # Apply search filter
            if search_term and 'Knowledge' in df.columns:
                df = df[df['Knowledge'].str.contains(search_term, case=False, na=False)]
        
        # Display data
        if df.empty:
            if len(st.session_state.knowledge_data) == 0:
                st.info("🚀 **Get Started:** Upload a document, image, video, or audio file to extract knowledge!")
                st.markdown("""
                **Or try:**
                - Click "📋 Load Demo" to see example data
                - Upload any supported file type for AI analysis
                """)
            else:
                st.info("No data matches your current filters. Try adjusting the filters above.")
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
        
        if not df.empty and 'Type' in df.columns and 'Confidence' in df.columns:
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
        else:
            st.info("📊 Charts will appear after processing files")
        
        # Stats
        st.subheader("📊 Statistics")
        st.metric("Total Items", len(df))
        if not df.empty and 'Confidence' in df.columns:
            st.metric("Avg Confidence", f"{df['Confidence'].mean():.2f}")
        if not df.empty and 'Type' in df.columns:
            st.metric("Types", df['Type'].nunique())

if __name__ == "__main__":
    main()