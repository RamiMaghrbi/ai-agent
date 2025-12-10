import os

from langchain_ollama import OllamaEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .loader import load_raw_texts

DEFAULT_DB_DIR = "chroma_db"
DEFAULT_DATA_DIR = "data"
EMBED_MODEL_NAME = "nomic-embed-text"
HF_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class SafeOllamaEmbeddings(OllamaEmbeddings):
    """Work around intermittent embed_documents failures by calling embed per text batch."""

    def embed_documents(self, texts):
        if not self._client:
            msg = (
                "Ollama client is not initialized. "
                "Please ensure Ollama is running and the model is loaded."
            )
            raise ValueError(msg)
        embeddings = []
        for text in texts:
            resp = self._client.embed(
                self.model,
                [text],
                options=self._default_params,
                keep_alive=self.keep_alive,
            )
            embeddings.append(resp["embeddings"][0])
        return embeddings

    def embed_query(self, text: str):
        return self.embed_documents([text])[0]


def get_embeddings():
    """
    Select embeddings backend.
    - DENTAL_EMBED_BACKEND=huggingface (default) uses sentence-transformers.
    - DENTAL_EMBED_BACKEND=ollama uses local Ollama embeddings.
    """
    backend = (os.getenv("DENTAL_EMBED_BACKEND") or "huggingface").lower()
    if backend == "huggingface":
        model_name = os.getenv("HUGGINGFACE_EMBED_MODEL", HF_EMBED_MODEL)
        return HuggingFaceEmbeddings(model_name=model_name)
    if backend == "ollama":
        base_url = os.getenv("OLLAMA_HOST") or "http://127.0.0.1:11434"
        return SafeOllamaEmbeddings(model=EMBED_MODEL_NAME, base_url=base_url)
    raise ValueError(f"Unsupported embedding backend: {backend}")


def build_vectorstore(
    data_dir: str = DEFAULT_DATA_DIR,
    db_dir: str = DEFAULT_DB_DIR,
) -> Chroma:
    texts = load_raw_texts(data_dir=data_dir)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
    )

    documents = []
    for content, path in texts:
        docs_for_file = splitter.create_documents(
            [content],
            metadatas=[{"source": path}],
        )
        documents.extend(docs_for_file)

    embeddings = get_embeddings()
    os.makedirs(db_dir, exist_ok=True)

    vectordb = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=db_dir,
    )
    return vectordb


def load_vectorstore(
    db_dir: str = DEFAULT_DB_DIR,
) -> Chroma:
    if not os.path.isdir(db_dir) or not os.listdir(db_dir):
        raise ValueError(
            f"ما في فهرس Chroma جاهز داخل '{db_dir}'. "
            "بدك تنادي build_vectorstore أول مرة."
        )

    embeddings = get_embeddings()

    vectordb = Chroma(
        embedding_function=embeddings,
        persist_directory=db_dir,
    )
    return vectordb


def get_or_create_vectorstore(
    data_dir: str = DEFAULT_DATA_DIR,
    db_dir: str = DEFAULT_DB_DIR,
    rebuild: bool = False,
) -> Chroma:
    if rebuild or not os.path.isdir(db_dir) or not os.listdir(db_dir):
        print("📚 عم نبني Chroma DB (dental-kb) من الصفر...")
        return build_vectorstore(data_dir=data_dir, db_dir=db_dir)

    print("✅ لقيت Chroma DB جاهزة، عم أحمّلها...")
    return load_vectorstore(db_dir=db_dir)
