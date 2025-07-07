import os
import shutil
import glob
from datetime import datetime
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings

# Disable telemetry warnings
os.environ["CHROMA_TELEMETRY_ENABLED"] = "false"

BASE_DIR = "./chroma_db"

def cleanup_old_sessions(current_session_path):
    if not os.path.exists(BASE_DIR):
        return

    folders = sorted(
        glob.glob(os.path.join(BASE_DIR, "session_*")),
        key=os.path.getmtime,
        reverse=True
    )

    for folder in folders:
        if folder == current_session_path:
            continue
        try:
            shutil.rmtree(folder)
        except Exception as e:
            print(f"Warning: Couldn't delete old session {folder}: {e}")

def embed_and_store(docs):
    os.makedirs(BASE_DIR, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = os.path.join(BASE_DIR, f"session_{timestamp}")

    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L12-v2")

    vectordb = Chroma.from_documents(
        documents=docs,
        embedding=embedding,
        persist_directory=session_dir
    )

    cleanup_old_sessions(current_session_path=session_dir)
    return vectordb, session_dir
