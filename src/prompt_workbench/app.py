"""Streamlit entry point — layout and wiring only.

One screen: pick a situation, read or edit the prompt it starts from, and talk.
Everything it could decide has been pushed into ``core`` so it can be tested
without a browser.
"""

import streamlit as st

from prompt_workbench import __version__
from prompt_workbench.ui import session, sidebar, workbench

st.set_page_config(page_title="Prompt Workbench", layout="wide")

st.title("Prompt Workbench")
st.caption(f"v{__version__}")

st.markdown(
    "Find the prompt that makes a job come out right. Pick a real situation — "
    "its knowledge base, candidate lists and tool schemas already mocked — then "
    "work on the prompt with an engineer, run it as the end user, and examine "
    "what comes back."
)

with st.sidebar:
    sidebar.render()

if not session.has_credentials():
    st.info(
        "Add a provider API key in the sidebar to enable the chat. Use cases and "
        "prompts stay readable and editable without a key."
    )

workbench.render()
