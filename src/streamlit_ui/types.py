from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Union


# Simple alias to document what a message can contain in Streamlit.
MessageContent = Union[str, Dict[str, str]]


@dataclass
class StreamlitMessage:
    """Lightweight message object used by the Streamlit chat renderer."""

    role: str  # "user" or "assistant"
    content: MessageContent
    metadata: Optional[Dict[str, Any]] = None


__all__ = ["MessageContent", "StreamlitMessage"]

