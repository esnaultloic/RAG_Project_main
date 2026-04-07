import os

# Workaround for protobuf/OpenTelemetry incompatibilities seen on Streamlit Cloud.
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import argparse
import re
import time
import unicodedata
from pathlib import Path
from llama_index.core import Document, VectorStoreIndex, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.embeddings import BaseEmbedding
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb
from dotenv import load_dotenv
from google import genai
from pypdf import PdfReader

# Chargement des variables d'environnement depuis le fichier .env
load_dotenv()

# --- Configuration ---
# La clé API peut être fournie via GEMINI_API_KEY ou GOOGLE_API_KEY.
gemini_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
if not gemini_key:
    raise ValueError("GEMINI_API_KEY or GOOGLE_API_KEY environment variable not found.")

# Normaliser les variables pour éviter l'avertissement du SDK quand les deux sont définies.
os.environ["GOOGLE_API_KEY"] = gemini_key
os.environ.pop("GEMINI_API_KEY", None)

GENAI_CLIENT = genai.Client(api_key=gemini_key)
SETTINGS_INITIALIZED = False

# Définition des modèles à utiliser
LLM_MODEL = "gemini-2.5-flash"
EMBED_MODEL = "gemini-embedding-001"
COLLECTION_NAME = "cv_rag_collection_v2"
EMBED_MAX_RETRIES = 5
EMBED_RETRY_DELAY_SECONDS = 2

# Configuration du chunking pour une meilleure récupération sur les CV
Settings.node_parser = SentenceSplitter(chunk_size=800, chunk_overlap=120)

# Custom embedding class using Google GenAI directly
class GoogleGenAIDirectEmbedding(BaseEmbedding):
    """Direct embedding using google.genai."""
    
    model_config = {"extra": "allow"}  # Allow extra fields
    
    def __init__(self, model_name: str = "gemini-embedding-001", **kwargs):
        super().__init__(**kwargs)
        self.model_name = model_name

    def _embed_with_retry(self, contents):
        last_error = None
        for attempt in range(1, EMBED_MAX_RETRIES + 1):
            try:
                response = GENAI_CLIENT.models.embed_content(
                    model=self.model_name,
                    contents=contents,
                )
                embeddings = getattr(response, "embeddings", None) or []
                if not embeddings:
                    raise ValueError("No embedding returned by Google GenAI.")
                return embeddings
            except Exception as error:
                last_error = error
                status_code = getattr(error, "status_code", None)
                is_retryable = status_code in {429, 500, 503, 504}
                if not is_retryable or attempt == EMBED_MAX_RETRIES:
                    raise
                wait_seconds = EMBED_RETRY_DELAY_SECONDS * attempt
                print(
                    f"Embedding attempt {attempt}/{EMBED_MAX_RETRIES} failed with status {status_code}. "
                    f"Retrying in {wait_seconds}s..."
                )
                time.sleep(wait_seconds)
        raise last_error

    def _extract_embedding_values(self, text: str):
        embeddings = self._embed_with_retry(text)
        return embeddings[0].values

    def _get_text_embeddings(self, texts):
        embeddings = self._embed_with_retry(texts)
        if len(embeddings) != len(texts):
            raise ValueError("Embedding response size does not match request size.")
        return [embedding.values for embedding in embeddings]
    
    def _get_query_embedding(self, query: str):
        """Get embedding for a single query"""
        return self._extract_embedding_values(query)
    
    def _get_text_embedding(self, text: str):
        """Get embedding for a single text"""
        return self._extract_embedding_values(text)
    
    async def _aget_query_embedding(self, query: str):
        """Async version (same as sync for now)"""
        return self._get_query_embedding(query)
    
    async def _aget_text_embedding(self, text: str):
        """Async version (same as sync for now)"""
        return self._get_text_embedding(text)

def ensure_rag_settings():
    """Initialise les composants Gemini uniquement au moment où ils sont requis."""
    global SETTINGS_INITIALIZED
    if SETTINGS_INITIALIZED:
        return
    Settings.llm = GoogleGenAI(model=LLM_MODEL, api_key=gemini_key)
    Settings.embed_model = GoogleGenAIDirectEmbedding(model_name=EMBED_MODEL)
    SETTINGS_INITIALIZED = True

# Configuration ChromaDB
db = chromadb.PersistentClient(path="./chroma_db")  # Stocke la base de données localement dans un dossier
CHROMA_COLLECTION_REF = None
RAW_DOCUMENTS_CACHE = None
RAW_CHUNKS_CACHE = None

SEARCH_STOPWORDS = {
    "a", "ai", "au", "aux", "avec", "ce", "ces", "dans", "de", "des", "du", "en",
    "et", "est", "la", "le", "les", "leur", "loic", "l", "mon", "par", "pour",
    "que", "qu", "qui", "ses", "son", "sur", "une", "utilise", "utilisee", "utilises",
    "utilisees", "quelles", "quelle", "quels"
}

TOOL_PATTERNS = {
    "expériences professionnelles": [
        ("python", "Python"),
        ("power bi", "Power BI"),
        ("ansible", "Ansible"),
        ("jenkins", "Jenkins"),
        ("elk", "ELK"),
        ("shapely", "Shapely"),
    ],
    "formation": [
        ("aws", "AWS"),
        ("hadoop", "Hadoop"),
        ("scikit-learn", "Scikit-learn"),
        ("tensorflow", "TensorFlow"),
        ("pytorch", "PyTorch"),
        ("python", "Python"),
        ("java", "Java"),
        ("sql", "SQL"),
        ("mysql", "MySQL"),
        ("power bi", "Power BI"),
        ("matplotlib", "Matplotlib"),
        ("plotly", "Plotly"),
        ("seaborn", "Seaborn"),
        ("r", "R"),
        ("numpy", "NumPy"),
        ("pandas", "Pandas"),
        ("rvest", "rvest"),
    ],
    "certification": [
        ("foundation models", "Foundation Models"),
        ("llms", "LLMs"),
        ("prompt engineering", "Prompt Engineering"),
        ("prompt chaining", "Prompt chaining"),
        ("rag", "RAG"),
        ("vertex ai", "Vertex AI"),
        ("mlops", "MLOps"),
        ("gemini", "Gemini"),
        ("document ai", "Document AI"),
        ("vision", "Vision"),
        ("speech-to-text", "Speech-to-Text"),
        ("secure ai framework", "Secure AI Framework"),
    ],
}


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return text.lower()


def tokenize_search_text(text: str):
    normalized = normalize_text(text)
    tokens = re.findall(r"[a-z0-9]+", normalized)
    return [token for token in tokens if len(token) > 1 and token not in SEARCH_STOPWORDS]


def load_source_documents(data_dir: str = "data"):
    global RAW_DOCUMENTS_CACHE
    if RAW_DOCUMENTS_CACHE is None:
        documents = []
        for path in sorted(Path(data_dir).iterdir()):
            if not path.is_file():
                continue

            if path.suffix.lower() == ".pdf":
                reader = PdfReader(str(path))
                page_texts = [(page.extract_text() or "").strip() for page in reader.pages]
                text = "\n\n".join(page_text for page_text in page_texts if page_text)
            else:
                text = path.read_text(encoding="utf-8", errors="ignore").strip()

            if not text:
                continue

            documents.append(
                Document(
                    text=text,
                    metadata={
                        "file_name": path.name,
                        "source": str(path),
                    },
                )
            )

        RAW_DOCUMENTS_CACHE = documents
    return RAW_DOCUMENTS_CACHE


def get_structured_tool_sections(data_dir: str = "data"):
    documents = load_source_documents(data_dir=data_dir)
    sections = {
        "expériences professionnelles": "",
        "formation": "",
        "certification": "",
    }

    for document in documents:
        source = document.metadata.get("file_name", "")
        normalized = normalize_text(document.text).replace("\x00", "")

        if source == "my_cv.pdf":
            start = normalized.find("experiences professionnelles")
            end = normalized.find("formations")
            if start != -1:
                sections["expériences professionnelles"] = normalized[start:end if end != -1 else None]

        if source == "Complement_cv.pdf":
            formation_start = normalized.find("formation cy tech")
            if formation_start != -1:
                sections["formation"] = normalized[formation_start:]
                sections["certification"] = normalized[:formation_start]
            else:
                sections["certification"] = normalized

    return sections


def build_tools_answer(data_dir: str = "data"):
    sections = get_structured_tool_sections(data_dir=data_dir)
    lines = []

    for category, patterns in TOOL_PATTERNS.items():
        found_tools = []
        section_text = sections.get(category, "")
        for needle, display_name in patterns:
            if needle in section_text and display_name not in found_tools:
                found_tools.append(display_name)

        if found_tools:
            lines.append(f"{category.capitalize()} : {', '.join(found_tools)}.")
        else:
            lines.append(f"{category.capitalize()} : non précisé dans les documents.")

    return "\n".join(lines)


def build_search_chunks(data_dir: str = "data"):
    global RAW_CHUNKS_CACHE
    if RAW_CHUNKS_CACHE is not None:
        return RAW_CHUNKS_CACHE

    documents = load_source_documents(data_dir=data_dir)
    splitter = SentenceSplitter(chunk_size=700, chunk_overlap=80)
    nodes = splitter.get_nodes_from_documents(documents)
    chunks = []
    for node in nodes:
        text = node.get_content().strip()
        if not text:
            continue
        metadata = getattr(node, "metadata", {}) or {}
        source = metadata.get("file_name") or metadata.get("filename") or metadata.get("source") or "document"
        chunks.append({
            "text": text,
            "source": source,
            "normalized": normalize_text(text),
            "tokens": set(tokenize_search_text(text)),
        })
    RAW_CHUNKS_CACHE = chunks
    return RAW_CHUNKS_CACHE


def keyword_retrieve(prompt: str, data_dir: str = "data", top_k: int = 6):
    query_tokens = tokenize_search_text(prompt)
    if not query_tokens:
        return []

    query_normalized = normalize_text(prompt)
    ranked_chunks = []
    for chunk in build_search_chunks(data_dir=data_dir):
        overlap = chunk["tokens"].intersection(query_tokens)
        if not overlap:
            continue

        score = len(overlap) * 10
        text_normalized = chunk["normalized"]
        for token in query_tokens:
            if token in text_normalized:
                score += 2

        if "outil" in query_normalized and "outils" in text_normalized:
            score += 20
        if "experience" in query_normalized and "experience" in text_normalized:
            score += 10
        if "formation" in query_normalized and "formation" in text_normalized:
            score += 10
        if "certification" in query_normalized and "certification" in text_normalized:
            score += 10

        ranked_chunks.append((score, chunk))

    ranked_chunks.sort(key=lambda item: item[0], reverse=True)
    return [chunk for _, chunk in ranked_chunks[:top_k]]


def collect_retrieved_contexts(prompt: str, index=None, similarity_top_k: int = 5, data_dir: str = "data"):
    contexts = []
    seen_texts = set()

    if index is not None:
        try:
            retrieved_nodes = index.as_retriever(similarity_top_k=similarity_top_k).retrieve(prompt)
            for node in retrieved_nodes:
                text = node.get_content().strip()
                if not text or text in seen_texts:
                    continue
                metadata = getattr(node, "metadata", {}) or {}
                contexts.append({
                    "text": text,
                    "source": metadata.get("file_name") or metadata.get("filename") or metadata.get("source") or "vector_store",
                })
                seen_texts.add(text)
        except Exception as error:
            print(f"Vector retrieval fallback triggered: {error}")

    for chunk in keyword_retrieve(prompt, data_dir=data_dir, top_k=max(similarity_top_k, 6)):
        text = chunk["text"]
        if text in seen_texts:
            continue
        contexts.append({"text": text, "source": chunk["source"]})
        seen_texts.add(text)

    return contexts


def synthesize_answer(prompt: str, contexts):
    if not contexts:
        return "Les documents disponibles ne contiennent pas d'information pertinente pour répondre à cette question."

    context_block = "\n\n".join(
        f"[Source: {context['source']}]\n{context['text']}" for context in contexts
    )
    synthesis_prompt = f"""
Tu es un assistant RH qui répond uniquement à partir du contexte fourni.

Consignes:
- Réponds en français.
- N'invente rien.
- Si la question porte sur des outils, liste explicitement les outils trouvés.
- Quand c'est possible, regroupe la réponse par catégories pertinentes comme expériences professionnelles, formation et certification.
- Si une catégorie n'apparaît pas dans le contexte, indique simplement qu'elle n'est pas précisée.

Question:
{prompt}

Contexte:
{context_block}
""".strip()
    response = Settings.llm.complete(synthesis_prompt)
    return str(response)

def setup_rag_index(data_dir: str = "data", force_reindex: bool = False):
    """
    Charge les documents du répertoire data et crée/charge l'index RAG.
    """
    rebuilding_collection = False
    try:
        ensure_rag_settings()
        # Récupération de la liste des noms de collections existantes
        existing_collections = [c.name for c in db.list_collections()]
        
        # Vérification si la collection existe déjà
        if COLLECTION_NAME in existing_collections and not force_reindex:
            print(f"Loading existing collection: {COLLECTION_NAME}")
            # Utilisation de get_collection pour récupérer l'objet collection existant
            chroma_collection = db.get_collection(COLLECTION_NAME)
            globals()["CHROMA_COLLECTION_REF"] = chroma_collection
            # Validation que la collection contient des embeddings/documents
            try:
                count = chroma_collection.count()
            except Exception:
                count = 0
            if count == 0:
                print(f"La collection existante '{COLLECTION_NAME}' est vide. Reconstruction de l'index...")
                rebuilding_collection = True
                db.delete_collection(COLLECTION_NAME)
                chroma_collection = db.get_or_create_collection(COLLECTION_NAME)
                globals()["CHROMA_COLLECTION_REF"] = chroma_collection
                vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
                print(f"Loading documents from {data_dir}...")
                documents = load_source_documents(data_dir=data_dir)
                print(f"Creating index from {len(documents)} documents...")
                index = VectorStoreIndex.from_documents(documents, vector_store=vector_store)
                print("Index created successfully")
            else:
                print(f"Chargement de la collection existante: {COLLECTION_NAME} (documents: {count})")
                vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
                index = VectorStoreIndex.from_vector_store(vector_store)
            
        else:
            if force_reindex and COLLECTION_NAME in existing_collections:
                print(f"Réindexation forcée activée. Suppression de la collection existante: {COLLECTION_NAME}")
                db.delete_collection(COLLECTION_NAME)
            rebuilding_collection = True
            print(f"Création et indexation de la nouvelle collection: {COLLECTION_NAME}")
            
            # 1. Chargement des données
            print(f"Loading documents from {data_dir}...")
            documents = load_source_documents(data_dir=data_dir)
            print(f"Loaded {len(documents)} documents")
            
            # 2. Configuration du Vector Store ChromaDB
            # Utilisation de get_or_create_collection pour robustesse
            print("Creating Chroma collection...")
            chroma_collection = db.get_or_create_collection(COLLECTION_NAME)
            globals()["CHROMA_COLLECTION_REF"] = chroma_collection
            vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
            
            # 3. Création de l'index
            print("Creating VectorStoreIndex from documents...")
            index = VectorStoreIndex.from_documents(
                documents,
                vector_store=vector_store
            )
            print("Index created successfully")
            
        # Retour de l'objet index, qu'il ait été chargé ou créé
        return index

    except Exception as e:
        if rebuilding_collection:
            try:
                db.delete_collection(COLLECTION_NAME)
            except Exception:
                pass
        print(f"Une erreur s'est produite lors de la configuration de l'index: {e}")
        import traceback
        traceback.print_exc()
        return None


def get_collection_count() -> int:
    """Retourne le nombre d'éléments stockés dans la collection Chroma.
    Essaie count(); si zéro, utilise un fallback en récupérant les ids pour éviter les problèmes de driver.
    """
    try:
        chroma_collection = globals().get("CHROMA_COLLECTION_REF")
        if chroma_collection is None:
            chroma_collection = db.get_or_create_collection(COLLECTION_NAME)
            globals()["CHROMA_COLLECTION_REF"] = chroma_collection

        try:
            cnt = int(chroma_collection.count())
        except Exception:
            cnt = 0
        if cnt > 0:
            return cnt
        # Méthodes de secours
        try:
            peeked = chroma_collection.peek(limit=10000)
            ids = peeked.get("ids") or []
            if ids:
                return len(ids)
        except Exception:
            pass
        try:
            got = chroma_collection.get(ids=None, where={}, limit=None)
            ids = got.get("ids") or []
            return len(ids)
        except Exception:
            return 0
    except Exception:
        return 0

# Création du répertoire 'data' s'il n'existe pas
os.makedirs("data", exist_ok=True)
print("Placez votre fichier CV (ex: my_cv.pdf) dans le dossier 'data'.")

# L'index est configuré une fois au démarrage de l'application (peut être surchargé par CLI)
RAG_INDEX = None

def query_rag(prompt: str, show_context: bool = True, index=None, similarity_top_k: int = 5) -> str:
    """
    Interroge l'index RAG et obtient une réponse ancrée dans le document CV.
    
    :param prompt: La question de l'utilisateur.
    :return: La réponse du LLM ancrée dans le document.
    """
    if "outil" in normalize_text(prompt):
        return build_tools_answer()

    # Préférer l'index fourni; sinon utiliser le global; en dernier recours, essayer d'initialiser
    ensure_rag_settings()
    active_index = index or RAG_INDEX
    if active_index is None:
        active_index = setup_rag_index()
    if active_index is None:
        return "Le système RAG a échoué à s'initialiser. Veuillez vérifier la configuration et la clé API."
    
    try:
        contexts = collect_retrieved_contexts(
            prompt,
            index=active_index,
            similarity_top_k=similarity_top_k,
        )
        if show_context:
            print("\n--- Chunks récupérés (debug) ---")
            for i, context in enumerate(contexts, start=1):
                print(f"[{i}] source={context['source']}\n{context['text'][:600]}\n")
            print("--- Fin des chunks récupérés ---\n")
        return synthesize_answer(prompt, contexts)
    except Exception as e:
        return f"Une erreur s'est produite pendant la requête: {e}"

# --- Exemple d'utilisation (Auto-test) ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Assistant RAG CV")
    parser.add_argument("--reindex", action="store_true", help="Forcer la réindexation de la collection Chroma")
    parser.add_argument("--question", type=str, default="Quel est mon dernier poste et quelles étaient mes principales responsabilités ?", help="Question à poser au RAG")
    args = parser.parse_args()

    # Initialisation de l'index avec réindexation forcée optionnelle
    RAG_INDEX = setup_rag_index(force_reindex=args.reindex)

    if RAG_INDEX:
        test_prompt = args.question
        print(f"\n[Requête de test]: {test_prompt}")
        response = query_rag(test_prompt, show_context=True)
        print("\n" + "="*50)
        print(f"[Réponse RAG]:\n{response}")
        print("="*50)
    else:
        print("\nÉchec de la configuration du pipeline. Impossible d'exécuter la requête de test.")
