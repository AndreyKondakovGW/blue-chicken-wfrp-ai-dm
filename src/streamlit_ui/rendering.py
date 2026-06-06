from __future__ import annotations

from typing import List

from .types import MessageContent, StreamlitMessage

# Default avatars for chat messages. The agent avatar path can be overridden by
# the top‑level runner if it wants to use a custom image instead of an emoji.
AGENT_AVATAR_DEFAULT = "🤖"
USER_AVATAR = "👤"


def _render_content(content: MessageContent) -> None:
    """Render one message body in Streamlit, handling rich types when possible."""
    import streamlit as st

    if isinstance(content, dict):
        path = content.get("path")
        mime = (content.get("mime_type") or "").lower()
        if path and ("image/" in mime or path.lower().endswith((".png", ".jpg", ".jpeg", ".webp", ".gif"))):
            st.image(path)
            return
        if path and ("audio/" in mime or path.lower().endswith((".wav", ".mp3", ".ogg", ".m4a"))):
            st.audio(path)
            return
        st.write(content)
        return

    # Allow small inline HTML (for footnotes), but keep everything else as markdown.
    if "<span" in content and "</span>" in content:
        st.markdown(content, unsafe_allow_html=True)
    else:
        st.markdown(content.replace("\n", "<br>"), unsafe_allow_html=True)


def render_messages(messages: List[StreamlitMessage], agent_avatar: str | None = None) -> None:
    """
    Render a linear list of `StreamlitMessage` objects as a Streamlit chat.

    Tool calls with metadata are grouped into expandable blocks, while regular
    assistant / user messages are shown as normal chat bubbles.
    """
    import streamlit as st

    avatar = agent_avatar or AGENT_AVATAR_DEFAULT

    i = 0
    while i < len(messages):
        msg = messages[i]
        meta = msg.metadata or {}

        if msg.role == "user":
            with st.chat_message("user", avatar=USER_AVATAR):
                _render_content(msg.content)
            i += 1
            continue

        # Assistant message
        title = meta.get("title")
        msg_id = meta.get("id")

        # Tool call parent message: group consecutive child messages with parent_id.
        if title and msg_id:
            with st.chat_message("assistant", avatar=avatar):
                with st.expander(title, expanded=False):
                    _render_content(msg.content)
                    j = i + 1
                    while j < len(messages):
                        child = messages[j]
                        child_meta = child.metadata or {}
                        if child_meta.get("parent_id") != msg_id:
                            break
                        child_title = child_meta.get("title")
                        if child_title:
                            st.markdown(f"**{child_title}**")
                        _render_content(child.content)
                        j += 1
            i = j
            continue

        # Child messages should have been rendered under a parent; fall back to normal rendering.
        with st.chat_message("assistant", avatar=avatar):
            if title:
                st.markdown(f"**{title}**")
            _render_content(msg.content)
        i += 1


__all__ = ["AGENT_AVATAR_DEFAULT", "USER_AVATAR", "render_messages"]

