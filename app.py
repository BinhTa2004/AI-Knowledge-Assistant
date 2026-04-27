import os
import hashlib
import tempfile
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from ultis.rag_core import answer_stream, build_rag_chain


@st.cache_resource(show_spinner="Initializing RAG, please wait...")
def get_cached_chain(source_dir_paths: tuple[str, ...]):
    source_dirs = [Path(path) for path in source_dir_paths]
    return build_rag_chain(source_dirs=source_dirs)


def save_uploaded_files(uploaded_files) -> Path | None:
    if not uploaded_files:
        return None

    digest = hashlib.sha256()
    for uploaded_file in uploaded_files:
        digest.update(uploaded_file.name.encode("utf-8"))
        digest.update(uploaded_file.getvalue())

    upload_root = Path(tempfile.gettempdir()) / "rag_chatbot_uploads" / digest.hexdigest()
    upload_root.mkdir(parents=True, exist_ok=True)

    for uploaded_file in uploaded_files:
        target_path = upload_root / uploaded_file.name
        target_path.write_bytes(uploaded_file.getvalue())

    return upload_root


def main():
    st.set_page_config(page_title="RAG Chat Demo", page_icon="💬", layout="wide")
    st.title("RAG Chat Demo")
    st.caption("Ask questions from your PDF documents with LangChain + Gemini")

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "active_source_dirs" not in st.session_state:
        st.session_state.active_source_dirs = None
    if "active_chain" not in st.session_state:
        st.session_state.active_chain = None
    if "upload_notice" not in st.session_state:
        st.session_state.upload_notice = None
    if "uploaded_file_names" not in st.session_state:
        st.session_state.uploaded_file_names = []

    default_data_dir = Path(__file__).resolve().parents[1] / "Rag ChatBot" / "data"
    default_source_dir = str(default_data_dir)

    load_dotenv()
    if not os.getenv("GOOGLE_API_KEY"):
        st.error("GOOGLE_API_KEY not found. Please add it to your .env file.")
        st.stop()

    with st.sidebar:
        st.subheader("Options")
        data_dir = st.text_input("Default data folder path", value=default_source_dir)
        uploaded_files = st.file_uploader(
            "Upload PDF files",
            type=["pdf"],
            accept_multiple_files=True,
            key="pdf_uploader",
        )
        if uploaded_files:
            st.caption("Selected files:")
            for uploaded_file in uploaded_files:
                st.write(f"- {uploaded_file.name}")

        load_button = st.button("Index uploaded files")
        if st.button("Clear chat history"):
            st.session_state.messages = []
            st.rerun()

    default_source_dirs = (str(Path(data_dir)),)

    if load_button:
        if not uploaded_files:
            st.session_state.upload_notice = "Please choose at least one PDF file first."
        else:
            try:
                with st.spinner("Saving, chunking, and embedding uploaded files..."):
                    upload_dir = save_uploaded_files(uploaded_files)
                    source_dirs = [str(Path(data_dir))]
                    if upload_dir is not None:
                        source_dirs.append(str(upload_dir))

                    rag_chain = get_cached_chain(tuple(source_dirs))

                st.session_state.active_source_dirs = tuple(source_dirs)
                st.session_state.active_chain = rag_chain
                st.session_state.upload_notice = "Files loaded successfully."
                st.session_state.uploaded_file_names = [uploaded_file.name for uploaded_file in uploaded_files]
                st.rerun()
            except Exception as exc:
                st.session_state.upload_notice = f"Failed to load files: {exc}"
                st.stop()

    if st.session_state.active_chain is not None:
        rag_chain = st.session_state.active_chain
    else:
        try:
            rag_chain = get_cached_chain(default_source_dirs)
        except Exception as exc:
            st.error(f"Failed to initialize RAG: {exc}")
            st.stop()

    if st.session_state.upload_notice:
        st.sidebar.success(st.session_state.upload_notice)

    if st.session_state.uploaded_file_names:
        st.sidebar.caption("Last indexed uploads:")
        for file_name in st.session_state.uploaded_file_names:
            st.sidebar.write(f"- {file_name}")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    user_input = st.chat_input("Type your question...")
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            try:
                response = st.write_stream(answer_stream(rag_chain, user_input))
            except Exception as exc:
                response = f"Error while generating answer: {exc}"
                st.error(response)

        st.session_state.messages.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    main()
