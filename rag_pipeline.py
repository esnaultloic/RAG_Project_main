import os

# Workaround for protobuf/OpenTelemetry incompatibilities seen on Streamlit Cloud.
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import argparse
import time
from pathlib import Path
from llama_index.core import Document, VectorStoreIndex, Settings, PromptTemplate
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
gemini_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
if not gemini_key:
    raise ValueError("GEMINI_API_KEY or GOOGLE_API_KEY environment variable not found.")

os.environ["GOOGLE_API_KEY"] = gemini_key
os.environ.pop("GEMINI_API_KEY", None)

GENAI_CLIENT = genai.Client(api_key=gemini_key)
SETTINGS_INITIALIZED = False

LLM_MODEL = "gemini-2.5-flash"
EMBED_MODEL = "gemini-embedding-001"
COLLECTION_NAME = "cv_rag_collection_v2"
EMBED_MAX_RETRIES = 5
EMBED_RETRY_DELAY_SECONDS = 2

Settings.node_parser = SentenceSplitter(chunk_size=512, chunk_overlap=64)

# Prompt qui encourage le LLM à citer explicitement les outils trouvés dans le contexte
QA_PROMPT_TMPL = (
    "Tu es un assistant RH. Réponds uniquement à partir des extraits du CV ci-dessous.\n"
    "- Réponds en français.\n"
    "- Si la question porte sur des outils ou technologies, liste-les explicitement tels qu'ils apparaissent dans le contexte.\n"
    "- Si une information n'est pas présente dans le contexte, dis-le clairement sans inventer.\n\n"
    "Extraits du CV :\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Question : {query_str}\n"
    "Réponse :"
)
QA_PROMPT = PromptTemplate(QA_PROMPT_TMPL)


class GoogleGenAIDirectEmbedding(BaseEmbedding):
    """Direct embedding using google.genai."""

    model_config = {"extra": "allow"}

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
                    f"Embedding attempt {attempt}/{EMBED_MAX_RETRIES} failed "
                    f"(status {status_code}). Retrying in {wait_seconds}s..."
                )
                time.sleep(wait_seconds)
        raise last_error

    def _extract_embedding_values(self, text: str):
        return self._embed_with_retry(text)[0].values

    def _get_text_embeddings(self, texts):
        embeddings = self._embed_with_retry(texts)
        if len(embeddings) != len(texts):
            raise ValueError("Embedding response size mismatch.")
        return [emb.values for emb in embeddings]

    def _get_query_embedding(self, query: str):
        return self._extract_embedding_values(query)

    def _get_text_embedding(self, text: str):
        return self._extract_embedding_values(text)

    async def _aget_query_embedding(self, query: str):
        return self._get_query_embedding(query)

    async def _aget_text_embedding(self, text: str):
        return self._get_text_embedding(text)


def ensure_rag_settings():
    global SETTINGS_INITIALIZED
    if SETTINGS_INITIALIZED:
        return
    Settings.llm = GoogleGenAI(model=LLM_MODEL, api_key=gemini_key)
    Settings.embed_model = GoogleGenAIDirectEmbedding(model_name=EMBED_MODEL)
    SETTINGS_INITIALIZED = True


db = chromadb.PersistentClient(path="./chroma_db")
CHROMA_COLLECTION_REF = None

RAW_DOCUMENTS_CACHE = None


def load_source_documents(data_dir: str = "data"):
    """Charge les PDF et fichiers texte du dossier data via pypdf."""
    global RAW_DOCUMENTS_CACHE
    if RAW_DOCUMENTS_CACHE is not None:
        return RAW_DOCUMENTS_CACHE

    documents = []
    for path in sorted(Path(data_dir).iterdir()):
        if not path.is_file():
            continue

        if path.suffix.lower() == ".pdf":
            reader = PdfReader(str(path))
            page_texts = [(page.extract_text() or "").strip() for page in reader.pages]
            text = "\n\n".join(p for p in page_texts if p)
        else:
            text = path.read_text(encoding="utf-8", errors="ignore").strip()

        if not text:
            continue

        documents.append(
            Document(
                text=text,
                metadata={"file_name": path.name, "source": str(path)},
            )
        )

    RAW_DOCUMENTS_CACHE = documents
    return RAW_DOCUMENTS_CACHE


def setup_rag_index(data_dir: str = "data", force_reindex: bool = False):
    """Charge les documents et crée/charge l'index RAG vectoriel."""
    rebuilding_collection = False
    try:
        ensure_rag_settings()
        existing_collections = [c.name for c in db.list_collections()]

        if COLLECTION_NAME in existing_collections and not force_reindex:
            print(f"Loading existing collection: {COLLECTION_NAME}")
            chroma_collection = db.get_collection(COLLECTION_NAME)
            globals()["CHROMA_COLLECTION_REF"] = chroma_collection
            try:
                count = chroma_collection.count()
            except Exception:
                count = 0

            if count == 0:
                print("Collection vide, reconstruction...")
                rebuilding_collection = True
                db.delete_collection(COLLECTION_NAME)
                chroma_collection = db.get_or_create_collection(COLLECTION_NAME)
                globals()["CHROMA_COLLECTION_REF"] = chroma_collection
                vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
                documents = load_source_documents(data_dir=data_dir)
                print(f"Indexing {len(documents)} documents...")
                index = VectorStoreIndex.from_documents(documents, vector_store=vector_store)
                print("Index created successfully")
            else:
                print(f"Collection loaded: {count} chunks")
                vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
                index = VectorStoreIndex.from_vector_store(vector_store)

        else:
            if force_reindex and COLLECTION_NAME in existing_collections:
                print("Suppression de l'ancienne collection...")
                db.delete_collection(COLLECTION_NAME)
            rebuilding_collection = True
            print(f"Creating collection: {COLLECTION_NAME}")
            documents = load_source_documents(data_dir=data_dir)
            print(f"Loaded {len(documents)} documents")
            chroma_collection = db.get_or_create_collection(COLLECTION_NAME)
            globals()["CHROMA_COLLECTION_REF"] = chroma_collection
            vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
            print("Creating VectorStoreIndex from documents...")
            index = VectorStoreIndex.from_documents(documents, vector_store=vector_store)
            print("Index created successfully")

        return index

    except Exception as e:
        if rebuilding_collection:
            try:
                db.delete_collection(COLLECTION_NAME)
            except Exception:
                pass
        print(f"Erreur lors de la configuration de l'index: {e}")
        import traceback
        traceback.print_exc()
        return None


def get_collection_count() -> int:
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
        try:
            peeked = chroma_collection.peek(limit=10000)
            ids = peeked.get("ids") or []
            if ids:
                return len(ids)
        except Exception:
            pass
        return 0
    except Exception:
        return 0


os.makedirs("data", exist_ok=True)

RAG_INDEX = None


def query_rag(prompt: str, show_context: bool = False, index=None, similarity_top_k: int = 5) -> str:
    """Interroge l'index RAG vectoriel et retourne une réponse ancrée dans le CV."""
    ensure_rag_settings()
    active_index = index or RAG_INDEX
    if active_index is None:
        active_index = setup_rag_index()
    if active_index is None:
        return "Le système RAG a échoué à s'initialiser. Vérifiez la clé API."

    query_engine = active_index.as_query_engine(
        similarity_top_k=similarity_top_k,
        text_qa_template=QA_PROMPT,
    )

    try:
        response = query_engine.query(prompt)
        if show_context:
            print("\n--- Chunks récupérés (debug) ---")
            for i, node in enumerate(getattr(response, "source_nodes", []) or [], start=1):
                text = getattr(node, "text", "")
                score = getattr(node, "score", None)
                metadata = getattr(node, "metadata", {}) or {}
                source = metadata.get("file_name", "?")
                print(f"[{i}] source={source} score={score}\n{text[:600]}\n")
            print("--- Fin des chunks ---\n")
        return str(response)
    except Exception as e:
        return f"Erreur pendant la requête : {e}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Assistant RAG CV")
    parser.add_argument("--reindex", action="store_true", help="Forcer la réindexation")
    parser.add_argument(
        "--question",
        type=str,
        default="Quelles sont les outils que Loïc a utilisé dans ses expériences professionnelles, sa formation et sa certification ?",
    )
    args = parser.parse_args()

    RAG_INDEX = setup_rag_index(force_reindex=args.reindex)

    if RAG_INDEX:
        print(f"\n[Requête]: {args.question}")
        response = query_rag(args.question, show_context=True, index=RAG_INDEX)
        print("\n" + "=" * 50)
        print(f"[Réponse RAG]:\n{response}")
        print("=" * 50)
    else:
        print("\nÉchec de l'initialisation du pipeline.")
