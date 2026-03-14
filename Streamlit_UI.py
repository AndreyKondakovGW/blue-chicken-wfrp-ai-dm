#!/usr/bin/env python
# coding=utf-8
"""
Thin entry point for the Streamlit UI.

The original, larger implementation has been split into smaller, focused
modules under `src/streamlit_ui/`:

- `types.py` and `agent_stream.py` – interaction with the smolagents engine
- `rendering.py` – chat rendering helpers
- `styling.py` – layout / CSS and asset handling
- `app.py` – `StreamlitUI` class and `main()` entry point

You can still launch the app with:

    streamlit run Streamlit_UI.py
"""

from src.streamlit_ui.agent_stream import StreamlitMessage, stream_to_streamlit_messages
from src.streamlit_ui.app import StreamlitUI, main as _main

__all__ = ["StreamlitUI", "stream_to_streamlit_messages", "StreamlitMessage", "_main"]

if __name__ == "__main__":
    # When run via `streamlit run Streamlit_UI.py`, Streamlit executes this
    # file as a script, so this guard is triggered and we delegate to the
    # real `main()` implementation in `src.streamlit_ui.app`.
    _main()