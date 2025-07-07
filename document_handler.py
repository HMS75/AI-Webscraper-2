# document_handler.py
from langchain.document_loaders import UnstructuredFileLoader

def load_documents(file_paths):
    all_docs = []
    for path in file_paths:
        loader = UnstructuredFileLoader(path)
        docs = loader.load()
        all_docs.extend(docs)
    return all_docs
