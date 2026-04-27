from dataclasses import dataclass

MARKDOWN_SEPARATORS = [
    "\\n#{1,6} ",
    "```\\n",
    "\\n\\\\*\\\\*\\\\*+\\n",
    "\\n---+\\n",
    "\\n___+\\n",
    "\\n\\n",
    "\\n",
    " ",
    "",
]

DEFAULT_PROMPT_TEMPLATE = (
    "You are a strict, citation-focused assistant for a private knowledge base. \\n"
    "RULES:\\n"
    "1) Use ONLY the provided context to answer. \\n"
    "2) If the answer is not clearly contained in the context, say: "
    "\\\"I don't know based on the provided documents.\\\"\\n"
    "3) Do NOT use outside knowledge, guessing, or web information. \\n"
    "4) If applicable, cite sources as (source:page) using the metadata. \\n\\n"
    "Context:\\n{context}\\n\\n"
    "Question: {question}"
)


@dataclass(frozen=True)
class RagConfig:
    chunk_size: int = 1200
    chunk_overlap: int = 200
    embedding_model: str = "gemini-embedding-2-preview"
    embedding_dimensions: int = 1536
    llm_model: str = "models/gemini-2.5-flash"
    temperature: float = 0.0
    search_k: int = 5
    score_threshold: float = 0.2
    prompt_template: str = DEFAULT_PROMPT_TEMPLATE
