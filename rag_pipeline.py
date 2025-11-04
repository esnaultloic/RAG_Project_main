import os
import argparse
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore # <-- Make sure this is present
import chromadb
from dotenv import load_dotenv # <--- New Import

# Load environment variables from .env file
load_dotenv() # <--- New line at the start of the script

# --- Configuration ---
# Your Gemini API Key must be set as an environment variable (GEMINI_API_KEY)
gemini_key = os.getenv("GEMINI_API_KEY")
if not gemini_key:
    raise ValueError("GEMINI_API_KEY environment variable not found.")

# LlamaIndex's GoogleGenAI expects GOOGLE_API_KEY if not provided explicitly. Ensure mapping.
os.environ.setdefault("GOOGLE_API_KEY", gemini_key)

# Define the models we want to use
LLM_MODEL = "gemini-2.5-flash"
EMBED_MODEL = "text-embedding-004"
COLLECTION_NAME = "cv_rag_collection"

# Configure chunking for better retrieval on CVs
Settings.node_parser = SentenceSplitter(chunk_size=800, chunk_overlap=120)

# Initialize the LLM and Embedding Model (pass key explicitly for robustness)
Settings.llm = GoogleGenAI(model=LLM_MODEL, api_key=gemini_key)
Settings.embed_model = GoogleGenAIEmbedding(model=EMBED_MODEL, api_key=gemini_key)

# ChromaDB Setup
db = chromadb.PersistentClient(path="./chroma_db") # Stores the database locally in a folder
CHROMA_COLLECTION_REF = None

def setup_rag_index(data_dir: str = "data", force_reindex: bool = False):
    """
    Loads documents from the data directory and creates/loads the RAG index.
    """
    try:
        # Get the list of existing collection names
        existing_collections = [c.name for c in db.list_collections()]
        
        # Check if the collection already exists
        if COLLECTION_NAME in existing_collections and not force_reindex:
            # Use get_collection to retrieve the existing collection object
            chroma_collection = db.get_collection(COLLECTION_NAME)
            globals()["CHROMA_COLLECTION_REF"] = chroma_collection
            # Validate collection has embeddings/documents
            try:
                count = chroma_collection.count()
            except Exception:
                count = 0
            if count == 0:
                print(f"Existing collection '{COLLECTION_NAME}' is empty. Rebuilding index...")
                db.delete_collection(COLLECTION_NAME)
                chroma_collection = db.get_or_create_collection(COLLECTION_NAME)
                globals()["CHROMA_COLLECTION_REF"] = chroma_collection
                vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
                documents = SimpleDirectoryReader(input_dir=data_dir).load_data()
                index = VectorStoreIndex.from_documents(documents, vector_store=vector_store)
            else:
                print(f"Loading existing collection: {COLLECTION_NAME} (documents: {count})")
                vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
                index = VectorStoreIndex.from_vector_store(vector_store)
            
        else:
            if force_reindex and COLLECTION_NAME in existing_collections:
                print(f"Force reindex enabled. Deleting existing collection: {COLLECTION_NAME}")
                db.delete_collection(COLLECTION_NAME)
            print(f"Creating and indexing new collection: {COLLECTION_NAME}")
            
            # 1. Load Data
            documents = SimpleDirectoryReader(input_dir=data_dir).load_data()
            
            # 2. Setup ChromaDB Vector Store
            # Use get_or_create_collection for robustness
            chroma_collection = db.get_or_create_collection(COLLECTION_NAME)
            globals()["CHROMA_COLLECTION_REF"] = chroma_collection
            vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
            
            # 3. Create Index
            index = VectorStoreIndex.from_documents(
                documents,
                vector_store=vector_store
            )
            
        # Return the index object, whether it was loaded or created
        return index

    except Exception as e:
        print(f"An error occurred during index setup: {e}")
        return None


def get_collection_count() -> int:
    """Return number of items stored in the Chroma collection.
    Tries count(); if zero, falls back to fetching ids to avoid driver issues.
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
        # Fallbacks
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

# Create the 'data' directory if it doesn't exist
os.makedirs("data", exist_ok=True)
print("Place your CV file (e.g., my_cv.pdf) inside the 'data' folder.")

# The index is set up once when the application starts (may be overridden by CLI)
RAG_INDEX = None

def query_rag(prompt: str, show_context: bool = True, index=None, similarity_top_k: int = 5) -> str:
    """
    Queries the RAG index and gets a response grounded in the CV document.
    
    :param prompt: The user's question.
    :return: The LLM's grounded response.
    """
    # Prefer provided index; fallback to global; last resort, try to initialize
    active_index = index or RAG_INDEX
    if active_index is None:
        active_index = setup_rag_index()
    if active_index is None:
        return "RAG system failed to initialize. Please check the setup and API key."
    
    # Create the query engine from the index
    # We use 'as_query_engine' to handle the retrieval and generation automatically
    query_engine = active_index.as_query_engine(
        similarity_top_k=similarity_top_k,
    )
    
    try:
        response = query_engine.query(prompt)
        if show_context:
            print("\n--- Retrieved Chunks (debug) ---")
            for i, node in enumerate(getattr(response, "source_nodes", []) or [], start=1):
                text = getattr(node, "text", "")
                score = getattr(node, "score", None)
                print(f"[{i}] score={score}\n{text[:600]}\n")
            print("--- End Retrieved Chunks ---\n")
        return str(response)
    except Exception as e:
        return f"An error occurred during query: {e}"

# --- Example Usage (Self-Testing) ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG CV Assistant")
    parser.add_argument("--reindex", action="store_true", help="Force reindexing the Chroma collection")
    parser.add_argument("--question", type=str, default="Quel est mon dernier poste et quelles étaient mes principales responsabilités ?", help="Question à poser au RAG")
    args = parser.parse_args()

    # Initialize index with optional force reindex
    RAG_INDEX = setup_rag_index(force_reindex=args.reindex)

    if RAG_INDEX:
        test_prompt = args.question
        print(f"\n[Test Query]: {test_prompt}")
        response = query_rag(test_prompt, show_context=True)
        print("\n" + "="*50)
        print(f"[RAG Response]:\n{response}")
        print("="*50)
    else:
        print("\nPipeline setup failed. Cannot run test query.")