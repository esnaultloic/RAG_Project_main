import os

# Workaround for protobuf/OpenTelemetry incompatibilities seen on Streamlit Cloud.
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import argparse
import time
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.embeddings import BaseEmbedding
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb
from dotenv import load_dotenv
from google import genai

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
COLLECTION_NAME = "cv_rag_collection"
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
                documents = SimpleDirectoryReader(input_dir=data_dir).load_data()
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
            documents = SimpleDirectoryReader(input_dir=data_dir).load_data()
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
    # Préférer l'index fourni; sinon utiliser le global; en dernier recours, essayer d'initialiser
    ensure_rag_settings()
    active_index = index or RAG_INDEX
    if active_index is None:
        active_index = setup_rag_index()
    if active_index is None:
        return "Le système RAG a échoué à s'initialiser. Veuillez vérifier la configuration et la clé API."
    
    # Création du moteur de requête à partir de l'index
    # Utilisation de 'as_query_engine' pour gérer automatiquement la récupération et la génération
    query_engine = active_index.as_query_engine(
        similarity_top_k=similarity_top_k,
    )
    
    try:
        response = query_engine.query(prompt)
        if show_context:
            print("\n--- Chunks récupérés (debug) ---")
            for i, node in enumerate(getattr(response, "source_nodes", []) or [], start=1):
                text = getattr(node, "text", "")
                score = getattr(node, "score", None)
                print(f"[{i}] score={score}\n{text[:600]}\n")
            print("--- Fin des chunks récupérés ---\n")
        return str(response)
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
