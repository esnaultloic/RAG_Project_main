import os
import argparse
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.embeddings import BaseEmbedding
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb
from dotenv import load_dotenv
import google.genai as genai

# Chargement des variables d'environnement depuis le fichier .env
load_dotenv()

# --- Configuration ---
# La clé API Gemini doit être définie comme variable d'environnement (GEMINI_API_KEY)
gemini_key = os.getenv("GEMINI_API_KEY")
if not gemini_key:
    raise ValueError("GEMINI_API_KEY environment variable not found.")

# GoogleGenAI de LlamaIndex attend GOOGLE_API_KEY si non fournie explicitement. Assurer le mapping.
os.environ.setdefault("GOOGLE_API_KEY", gemini_key)

# Définition des modèles à utiliser
LLM_MODEL = "gemini-2.5-flash"
EMBED_MODEL = "models/embedding-001"  # Correct model name for Google API
COLLECTION_NAME = "cv_rag_collection"

# Configuration du chunking pour une meilleure récupération sur les CV
Settings.node_parser = SentenceSplitter(chunk_size=800, chunk_overlap=120)

# Custom embedding class using Google Generative AI directly (bypasses llama-index bug)
class GoogleGenAIDirectEmbedding(BaseEmbedding):
    """Direct embedding using google.genai to bypass llama-index model name bug"""
    
    model_config = {"extra": "allow"}  # Allow extra fields for api_key
    
    def __init__(self, model_name: str = "models/embedding-001", api_key: str = None, **kwargs):
        super().__init__(**kwargs)
        self.model_name = model_name
        # Store api_key in the object but Pydantic won't validate it
        object.__setattr__(self, '_api_key', api_key or os.getenv("GOOGLE_API_KEY"))
    
    def _get_query_embedding(self, query: str):
        """Get embedding for a single query"""
        api_key = getattr(self, '_api_key', None) or os.getenv("GOOGLE_API_KEY")
        result = genai.embed_content(
            model=self.model_name,
            content=query,
            api_key=api_key
        )
        return result['embedding']
    
    def _get_text_embedding(self, text: str):
        """Get embedding for a single text"""
        api_key = getattr(self, '_api_key', None) or os.getenv("GOOGLE_API_KEY")
        result = genai.embed_content(
            model=self.model_name,
            content=text,
            api_key=api_key
        )
        return result['embedding']
    
    async def _aget_query_embedding(self, query: str):
        """Async version (same as sync for now)"""
        return self._get_query_embedding(query)
    
    async def _aget_text_embedding(self, text: str):
        """Async version (same as sync for now)"""
        return self._get_text_embedding(text)

# Initialisation du LLM et du modèle d'embedding CUSTOM (bypass llama-index bug)
Settings.llm = GoogleGenAI(model=LLM_MODEL, api_key=gemini_key)
Settings.embed_model = GoogleGenAIDirectEmbedding(model_name=EMBED_MODEL, api_key=gemini_key)

# Configuration ChromaDB
db = chromadb.PersistentClient(path="./chroma_db")  # Stocke la base de données localement dans un dossier
CHROMA_COLLECTION_REF = None

def setup_rag_index(data_dir: str = "data", force_reindex: bool = False):
    """
    Charge les documents du répertoire data et crée/charge l'index RAG.
    """
    import logging
    logger = logging.getLogger(__name__)
    
    try:
        logger.info(f"Starting RAG index setup with embed_model={EMBED_MODEL}")
        
        # Récupération de la liste des noms de collections existantes
        existing_collections = [c.name for c in db.list_collections()]
        logger.info(f"Existing collections: {existing_collections}")
        
        # FORCE REINDEX: Delete old collection to ensure compatibility with new embedding model
        # This prevents using old embeddings with new code
        if COLLECTION_NAME in existing_collections:
            # Check if collection was created with old embedding model by trying to count
            try:
                chroma_collection_test = db.get_collection(COLLECTION_NAME)
                count_test = chroma_collection_test.count()
                if count_test > 0:
                    # Collection exists with data - delete it to force rebuild with new model
                    logger.info(f"⚠️ Deleting existing collection to rebuild with new embedding model: {COLLECTION_NAME}")
                    db.delete_collection(COLLECTION_NAME)
                    force_reindex = True
            except Exception as e:
                logger.warning(f"⚠️ Could not validate collection, rebuilding: {e}")
                try:
                    db.delete_collection(COLLECTION_NAME)
                    force_reindex = True
                except Exception as del_e:
                    logger.error(f"Could not delete collection: {del_e}")
        
        # Vérification si la collection existe déjà
        if COLLECTION_NAME in existing_collections and not force_reindex:
            logger.info(f"Loading existing collection: {COLLECTION_NAME}")
            # Utilisation de get_collection pour récupérer l'objet collection existant
            chroma_collection = db.get_collection(COLLECTION_NAME)
            globals()["CHROMA_COLLECTION_REF"] = chroma_collection
            # Validation que la collection contient des embeddings/documents
            try:
                count = chroma_collection.count()
            except Exception:
                count = 0
            if count == 0:
                logger.info(f"La collection existante '{COLLECTION_NAME}' est vide. Reconstruction de l'index...")
                db.delete_collection(COLLECTION_NAME)
                chroma_collection = db.get_or_create_collection(COLLECTION_NAME)
                globals()["CHROMA_COLLECTION_REF"] = chroma_collection
                vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
                logger.info(f"Loading documents from {data_dir}...")
                documents = SimpleDirectoryReader(input_dir=data_dir).load_data()
                logger.info(f"Creating index from {len(documents)} documents...")
                index = VectorStoreIndex.from_documents(documents, vector_store=vector_store)
                logger.info("Index created successfully")
            else:
                logger.info(f"Chargement de la collection existante: {COLLECTION_NAME} (documents: {count})")
                vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
                index = VectorStoreIndex.from_vector_store(vector_store)
            
        else:
            if force_reindex and COLLECTION_NAME in existing_collections:
                logger.info(f"Réindexation forcée activée. Suppression de la collection existante: {COLLECTION_NAME}")
                db.delete_collection(COLLECTION_NAME)
            logger.info(f"Création et indexation de la nouvelle collection: {COLLECTION_NAME}")
            
            # 1. Chargement des données
            logger.info(f"Loading documents from {data_dir}...")
            documents = SimpleDirectoryReader(input_dir=data_dir).load_data()
            logger.info(f"Loaded {len(documents)} documents")
            
            # 2. Configuration du Vector Store ChromaDB
            # Utilisation de get_or_create_collection pour robustesse
            logger.info("Creating Chroma collection...")
            chroma_collection = db.get_or_create_collection(COLLECTION_NAME)
            globals()["CHROMA_COLLECTION_REF"] = chroma_collection
            vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
            
            # 3. Création de l'index
            logger.info("Creating VectorStoreIndex from documents...")
            index = VectorStoreIndex.from_documents(
                documents,
                vector_store=vector_store
            )
            logger.info("Index created successfully")
            
        # Retour de l'objet index, qu'il ait été chargé ou créé
        return index

    except Exception as e:
        logger.error(f"Une erreur s'est produite lors de la configuration de l'index: {e}", exc_info=True)
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
