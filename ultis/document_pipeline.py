from pathlib import Path

from langchain_community.document_loaders import DirectoryLoader, UnstructuredFileLoader
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import MARKDOWN_SEPARATORS, RagConfig


def resolve_data_dir(data_dir: Path | None = None) -> Path:
    if data_dir is None:
        return Path(__file__).resolve().parents[2] / "Rag ChatBot" / "data"
    return data_dir


def load_documents(data_dir: Path):
    if not data_dir.exists():
        raise FileNotFoundError(f"Khong tim thay thu muc data: {data_dir}")

    loader = DirectoryLoader(
        str(data_dir),
        glob="**/*.pdf",
        show_progress=True,
        loader_cls=UnstructuredFileLoader,
        use_multithreading=True,
    )
    return loader.load()


def load_documents_from_sources(source_dirs: list[Path]):
    documents = []
    for source_dir in source_dirs:
        documents.extend(load_documents(source_dir))
    return documents


def split_documents(documents, config: RagConfig):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
        add_start_index=True,
        strip_whitespace=True,
        separators=MARKDOWN_SEPARATORS,
    )
    docs_splits = splitter.split_documents(documents)
    if not docs_splits:
        raise ValueError("Khong co noi dung nao duoc tach tu file PDF.")
    return docs_splits


def build_retriever(docs_splits, config: RagConfig):
    embeddings = GoogleGenerativeAIEmbeddings(
        model=config.embedding_model,
        output_dimensionality=config.embedding_dimensions,
    )

    texts = [doc.page_content for doc in docs_splits]
    metadatas = [doc.metadata for doc in docs_splits]

    # Embed each chunk to avoid potential batch-size mismatch with some backends.
    doc_embeddings = [embeddings.embed_query(text) for text in texts]

    vectorstore = FAISS.from_embeddings(
        text_embeddings=list(zip(texts, doc_embeddings)),
        embedding=embeddings,
        metadatas=metadatas,
        distance_strategy=DistanceStrategy.COSINE,
    )

    return vectorstore.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={
            "k": config.search_k,
            "score_threshold": config.score_threshold,
        },
    )
