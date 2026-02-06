import streamlit as st
import subprocess
import os

st.title("🔍 RAG Version Diagnostic")

try:
    # Get git commit
    commit = subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'],
        cwd=os.path.dirname(__file__),
        text=True
    ).strip()
    short_commit = commit[:8]
    
    # Import and display model config
    from rag_pipeline import EMBED_MODEL, LLM_MODEL
    
    st.success("✅ Code loaded successfully")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Commit (HEAD)", short_commit)
    with col2:
        st.metric("LLM Model", LLM_MODEL)
    with col3:
        st.metric("Embedding Model", EMBED_MODEL)
    
    st.info(f"Full commit: {commit}")
    
except Exception as e:
    st.error(f"Error loading version info: {e}")
    import traceback
    st.code(traceback.format_exc())
