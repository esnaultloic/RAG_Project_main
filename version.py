import subprocess
import os

def get_version_info():
    try:
        # Get current commit
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], 
                                       cwd=os.path.dirname(__file__),
                                       text=True).strip()
        # Get embed model from rag_pipeline
        from rag_pipeline import EMBED_MODEL
        return {
            "commit": commit[:8],
            "embed_model": EMBED_MODEL
        }
    except:
        return {"error": "Could not get version info"}

if __name__ == "__main__":
    info = get_version_info()
    print(f"Version Info: {info}")
