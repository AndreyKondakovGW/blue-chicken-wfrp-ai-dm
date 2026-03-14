from __future__ import annotations

import base64
from pathlib import Path
from typing import List, Optional


# We deliberately compute the project root relative to this file:
#   .../src/streamlit_ui/styling.py  -> project root is three levels up.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
ASSETS_DIR = PROJECT_ROOT / "assets"

# Configure asset file names here if you change them on disk.
PAGE_BACKGROUND_IMAGE = ASSETS_DIR / "page_bg.jpg"
CHAT_BACKGROUND_IMAGE = ASSETS_DIR / "chat_bg.jpg"
AGENT_AVATAR_IMAGE = ASSETS_DIR / "icon1.jpg"


def _load_image_base64(path: Path) -> Optional[str]:
    """Return a base64‑encoded string for an image file, or None if missing."""
    if not path.exists():
        return None
    data = path.read_bytes()
    return base64.b64encode(data).decode("utf-8")


def inject_global_css() -> None:
    """
    Inject global CSS for the Streamlit app.

    This includes:
    - full‑page background image (if available)
    - a centered "chat card" with its own background (image or color fallback)
    """
    import streamlit as st

    css_parts: List[str] = []

    # Full‑page background.
    page_b64 = _load_image_base64(PAGE_BACKGROUND_IMAGE)
    if page_b64:
        css_parts.append(
            f"""
            .stApp {{
                background: url("data:image/jpeg;base64,{page_b64}") no-repeat center center fixed;
                background-size: cover;
            }}
            """
        )

    # Central chat background block:
    # we insert a small marker div `.chat-wrapper` before the real chat container,
    # then style only the immediate sibling block.
    chat_b64 = _load_image_base64(CHAT_BACKGROUND_IMAGE)
    if chat_b64:
        css_parts.append(
            f"""
            .chat-wrapper {{
                max-width: 960px;
                margin: 1.5rem auto 0.25rem auto;
            }}
            .chat-wrapper + div[data-testid="stVerticalBlock"] {{
                max-width: 960px;
                margin: 0 auto 0 auto;
                background: url("data:image/png;base64,{chat_b64}") center center / cover no-repeat;
                padding: 1.5rem 1.75rem;
                border-radius: 18px;
                box-shadow: 0 4px 18px rgba(0, 0, 0, 0.55);
            }}
            """
        )
    else:
        # Fallback styling if no chat background image is present.
        css_parts.append(
            """
            .chat-wrapper {
                max-width: 960px;
                margin: 1.5rem auto 0.25rem auto;
            }
            .chat-wrapper + div[data-testid="stVerticalBlock"] {
                max-width: 960px;
                margin: 0 auto 0 auto;
                background-color: rgba(15, 17, 26, 0.9);
                padding: 1.5rem 1.75rem;
                border-radius: 18px;
                box-shadow: 0 4px 18px rgba(0, 0, 0, 0.55);
            }
            """
        )

    if css_parts:
        st.markdown("<style>" + "\n".join(css_parts) + "</style>", unsafe_allow_html=True)


__all__ = [
    "PROJECT_ROOT",
    "ASSETS_DIR",
    "PAGE_BACKGROUND_IMAGE",
    "CHAT_BACKGROUND_IMAGE",
    "AGENT_AVATAR_IMAGE",
    "inject_global_css",
]

