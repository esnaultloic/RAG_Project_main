import os
import streamlit as st
from dotenv import load_dotenv

# Load env early (for local runs)
load_dotenv()

st.set_page_config(page_title="Loïc Esnault — Data Scientist / AI Engineer", page_icon="📄", layout="centered")

st.title("Loïc Esnault — Data Scientist / AI Engineer")
st.caption("Posez une question sur mon profil. Les réponses sont ancrées dans mon CV indexé (RAG).")

# Import after env load so that rag_pipeline picks up keys
from rag_pipeline import setup_rag_index, query_rag, get_collection_count


@st.cache_resource(show_spinner=True)
def get_index():
    # Do not force reindex in app; persist existing collection
    return setup_rag_index()


index = get_index()
if index is None:
    st.error("RAG non initialisé. Vérifiez la clé API et la collection Chroma.")
    st.stop()

question = st.text_area(
    "Posez une question sur mon profil",
    value="Quelles sont les outils que Loïc a utilisé dans ses expériences professionnelles, sa formation et sa certification ?",
    height=100,
    placeholder="Ex: Quelles expériences ai-je en machine learning ?",
)

# Fixe top_k à 4 (large pour plusieurs documents)
FIXED_TOP_K = 4

ask = st.button("Poser la question")

if ask:
    if not question.strip():
        st.warning("Merci d'entrer une question.")
        st.stop()

    with st.spinner("Génération de la réponse..."):
        try:
            # Réponse simple, top_k fixé à 4
            answer = query_rag(question, show_context=False, index=index, similarity_top_k=FIXED_TOP_K)
            st.subheader("Réponse")
            st.write(answer)
        except Exception as e:
            st.error(f"Erreur pendant la requête: {e}")


