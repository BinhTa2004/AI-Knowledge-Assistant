from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI

from .config import RagConfig


def build_chain(retriever, config: RagConfig):
    prompt = ChatPromptTemplate.from_template(config.prompt_template)
    llm = ChatGoogleGenerativeAI(
        model=config.llm_model,
        temperature=config.temperature,
    )

    return (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
