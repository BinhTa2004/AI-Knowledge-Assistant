from pathlib import Path

from dotenv import load_dotenv
from .chain_factory import build_chain
from .config import RagConfig
from .document_pipeline import (
    build_retriever,
    load_documents_from_sources,
    resolve_data_dir,
    split_documents,
)


def build_rag_chain(
    data_dir: Path | None = None,
    config: RagConfig | None = None,
    source_dirs: list[Path] | None = None,
):
    load_dotenv()
    active_config = config or RagConfig()
    if source_dirs is None:
        resolved_data_dir = resolve_data_dir(data_dir)
        source_dirs = [resolved_data_dir]

    documents = load_documents_from_sources(source_dirs)
    docs_splits = split_documents(documents, active_config)
    retriever = build_retriever(docs_splits, active_config)
    return build_chain(retriever, active_config)

def answer_stream(chain, question: str):
    for chunk in chain.stream(question):
        if chunk:
            yield chunk
