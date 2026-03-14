from __future__ import annotations

from typing import Optional

from smolagents.agents import MultiStepAgent

from src.tools.rule_book import RuleBookTool

from .agent_stream import StreamlitMessage, stream_to_streamlit_messages
from .rendering import AGENT_AVATAR_DEFAULT, render_messages
from .styling import AGENT_AVATAR_IMAGE, PROJECT_ROOT, inject_global_css


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
        print("Rendering Streamlit UI...")
        import streamlit as st

        st.set_page_config(page_title="WFRP AI DM", layout="centered")
        self._ensure_state()
        inject_global_css()

        # Sidebar controls let the user choose between the full agent and
        # a direct rule‑book retrieval call, plus which rule book to use.
        mode = st.sidebar.radio(
            "Response mode",
            options=["Agent + tools", "Rule book retriever only"],
            key="response_mode",
        )
        rule_book_name = st.sidebar.selectbox(
            "Rule book",
            options=["wfrp_core_rulebook", "wfrp_up_in_arms_rulebook"],
            index=0,
            key="rule_book_name",
        )

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

        model = HfApiModel(
            max_tokens=2096,
            temperature=0.5,
            model_id="Qwen/Qwen2.5-Coder-32B-Instruct",
            custom_role_conversions=None,
        )

        prompts_path = PROJECT_ROOT / "prompts.yaml"
        with prompts_path.open("r", encoding="utf-8") as stream:
            prompt_templates = yaml.safe_load(stream)

        return CodeAgent(
            model=model,
            tools=[final_answer, rule_book_tool],
            max_steps=4,
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
    print("main")
    agent = build_default_agent()
    ui = StreamlitUI(agent)
    ui.render()


__all__ = ["StreamlitUI", "build_default_agent", "main"]

