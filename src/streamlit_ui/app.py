from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import List, Optional

from smolagents.agents import MultiStepAgent

from src.tools.rule_book import RuleBookTool
from src.tools.wfrpsu_itemlist import ItemListTool

from .agent_stream import StreamlitMessage, stream_to_streamlit_messages
from .rendering import AGENT_AVATAR_DEFAULT, render_messages
from .styling import AGENT_AVATAR_IMAGE, PROJECT_ROOT, inject_global_css


def _get_available_rule_books() -> List[str]:
    """Return names of vector stores that have both a FAISS index and a BM25 pickle."""
    databases_dir = PROJECT_ROOT / "databases"
    if not databases_dir.exists():
        return []
    books = []
    for entry in sorted(databases_dir.iterdir()):
        if entry.is_dir() and (entry / "index.faiss").exists():
            if (databases_dir / f"{entry.name}.pkl").exists():
                books.append(entry.name)
    return books


class StreamlitUI:
    """
    Streamlit chat UI for a `MultiStepAgent`.

    The UI can operate in two modes:
    - **Agent + tools** (default): run the full `CodeAgent` with all tools.
    - **Rule book retriever only**: bypass the agent and directly call
    `RuleBookTool.forward()` with the user question.

    Typical entry point from the project root:

        streamlit run Streamlit_UI.py
    """

    def __init__(self, agent: MultiStepAgent):
        self.agent = agent
        # Dedicated instance used when operating in "retriever‑only" mode.
        self.rule_book_tool = RuleBookTool()

    @staticmethod
    def _ensure_state() -> None:
        """Initialize Streamlit session state keys used by the chat."""
        import streamlit as st

        st.session_state.setdefault("messages", [])
        st.session_state.setdefault("file_uploads_log", [])
        st.session_state.setdefault("uploader_key", 0)

    @staticmethod
    def _append_user_prompt(prompt: str) -> str:
        """
        Optionally augment the user prompt with information about uploaded files.
        """
        import streamlit as st

        uploads = st.session_state.get("file_uploads_log", [])
        if uploads:
            return prompt + f"\nYou have been provided with these files, which might be helpful or not: {uploads}"
        return prompt

    def render(self) -> None:
        """Main render loop for the chat UI."""
        import streamlit as st

        st.set_page_config(page_title="WFRP AI DM", page_icon=":chicken:", layout="centered")
        
        self._ensure_state()
        inject_global_css()

        mode = st.sidebar.radio(
            "Response mode",
            options=["Agent + tools", "Rule book retriever only"],
            key="response_mode",
            index=1
        )

        available_books = _get_available_rule_books()
        rule_book_name = st.sidebar.selectbox(
            "Rule books",
            options=available_books if available_books else ["(no databases found)"],
            key="rule_book_name",
        )
        st.sidebar.subheader("Add new rule book")
        uploaded_pdf = st.sidebar.file_uploader(
            "Upload PDF",
            type=["pdf"],
            key=f"pdf_upload_{st.session_state['uploader_key']}",
        )
        if uploaded_pdf is not None:
            default_name = Path(uploaded_pdf.name).stem
            new_book_name = st.sidebar.text_input(
                "Vector store name", value=default_name, key="new_book_name"
            )
            if st.sidebar.button("Create vector store", type="primary"):
                name = new_book_name.strip()
                if not name:
                    st.sidebar.error("Please provide a vector store name.")
                else:
                    tmp_path: Optional[str] = None
                    try:
                        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                            tmp.write(uploaded_pdf.getvalue())
                            tmp_path = tmp.name

                        with st.status("Creating vector store…", expanded=True) as status:
                            from src.pdf_reader.hybrid_rag_search import (
                                HybridRetrival,
                                PDFReader,
                                OllamaEmbeddings,
                            )

                            st.write("Loading embedding model…")
                            embedder = OllamaEmbeddings(model="bge-m3")

                            st.write("Parsing PDF and splitting into chunks…")
                            pdf_reader = PDFReader()
                            chunks = pdf_reader.create_chunks_from_pdf(tmp_path, embedder)

                            st.write(f"Indexing {len(chunks)} chunks (dense + sparse)…")
                            db_path = str(PROJECT_ROOT / "databases")
                            retrival = HybridRetrival(
                                dense_embedder=embedder, database_path=db_path
                            )
                            retrival.create_new_vectorstore(name, chunks)

                            status.update(label="Done!", state="complete")

                        # Reset uploader and refresh the page so the new book appears.
                        st.session_state["uploader_key"] += 1
                        st.rerun()
                    except Exception as exc:
                        st.sidebar.error(f"Failed to create vector store: {exc}")
                    finally:
                        if tmp_path and os.path.exists(tmp_path):
                            os.unlink(tmp_path)

        st.title("Blue chicken WFRP AI DM")

        # The placeholder lets us re‑render the whole chat area as new messages arrive.
        chat_placeholder = st.empty()
        with chat_placeholder.container():
            # Marker div used only for CSS sibling selector to style the chat block.
            st.markdown('<div class="chat-wrapper">', unsafe_allow_html=True)
            render_messages(st.session_state["messages"], agent_avatar=self._agent_avatar())
            st.markdown("</div>", unsafe_allow_html=True)

        prompt = st.chat_input("Chat message")
        if not prompt:
            return

        # Add user message and re‑render immediately.
        st.session_state["messages"].append(StreamlitMessage(role="user", content=prompt))
        with chat_placeholder.container():
            st.markdown('<div class="chat-wrapper">', unsafe_allow_html=True)
            render_messages(st.session_state["messages"], agent_avatar=self._agent_avatar())
            st.markdown("</div>", unsafe_allow_html=True)

        if mode == "Rule book retriever only":
            # Direct retrieval: call the tool's `forward` method and show the
            # string response as a single assistant message.
            response_text = self.rule_book_tool.forward(query=prompt, rule_book_name=rule_book_name)
            st.session_state["messages"].append(StreamlitMessage(role="assistant", content=response_text))
            with chat_placeholder.container():
                st.markdown('<div class="chat-wrapper">', unsafe_allow_html=True)
                render_messages(st.session_state["messages"], agent_avatar=self._agent_avatar())
                st.markdown("</div>", unsafe_allow_html=True)
        else:
            # Full agent mode: augment the prompt (e.g. with uploaded files)
            # and stream messages as they are produced by the agent.
            augmented = self._append_user_prompt(prompt)
            for msg in stream_to_streamlit_messages(self.agent, task=augmented, reset_agent_memory=False):
                st.session_state["messages"].append(msg)
                with chat_placeholder.container():
                    st.markdown('<div class="chat-wrapper">', unsafe_allow_html=True)
                    render_messages(st.session_state["messages"], agent_avatar=self._agent_avatar())
                    st.markdown("</div>", unsafe_allow_html=True)

    @staticmethod
    def _agent_avatar() -> str:
        """Return either the custom agent avatar path or a default emoji."""
        return str(AGENT_AVATAR_IMAGE) if AGENT_AVATAR_IMAGE.exists() else AGENT_AVATAR_DEFAULT


def build_default_agent() -> MultiStepAgent:
    """
    Construct the default agent used by the Streamlit UI.

    This intentionally mirrors the setup in `app.py` but is wrapped in a
    `st.cache_resource` block so that the heavy initialization only happens
    once per Streamlit session.
    """
    import streamlit as st

    @st.cache_resource
    def _cached() -> MultiStepAgent:
        import yaml
        from smolagents import CodeAgent, HfApiModel
        from src.tools.final_answer import FinalAnswerTool
        from src.tools.rule_book import RuleBookTool

        final_answer = FinalAnswerTool()
        rule_book_tool = RuleBookTool()
        item_list_tool = ItemListTool()

        model = HfApiModel(
            max_tokens=2096,
            temperature=0.5,
            model_id="Qwen/Qwen2.5-Coder-32B-Instruct",
            custom_role_conversions=None,
        )

        prompts_path = PROJECT_ROOT / "prompts_ru.yaml"
        with prompts_path.open("r", encoding="utf-8") as stream:
            prompt_templates = yaml.safe_load(stream)

        return CodeAgent(
            model=model,
            tools=[final_answer, rule_book_tool, item_list_tool],
            max_steps=10,
            verbosity_level=1,
            grammar=None,
            planning_interval=None,
            name=None,
            description=None,
            prompt_templates=prompt_templates,
        )

    return _cached()


def main() -> None:
    """Entry point used by `streamlit run`."""
    agent = build_default_agent()
    ui = StreamlitUI(agent)
    ui.render()


__all__ = ["StreamlitUI", "build_default_agent", "main"]

