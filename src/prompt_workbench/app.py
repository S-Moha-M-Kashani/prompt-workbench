"""Streamlit entry point — layout and wiring only.

One page, walked top to bottom: describe the case, get test cases, write prompt
variants for that kind of job, choose the metrics, sweep them against models
cheapest-first, and keep the least expensive configuration that passed.

Everything this file could decide has been pushed into ``core`` so it can be
tested without a browser.
"""

import streamlit as st

from prompt_workbench import __version__
from prompt_workbench.ui import session, sidebar, workbench

st.set_page_config(page_title="Prompt Workbench", layout="wide")

st.title("Prompt Workbench")
st.caption(f"v{__version__}")

st.markdown(
    "Find a prompt, a model, settings and metrics you can defend — then write "
    "those metrics into the project that ships, where the same library will "
    "produce the same numbers."
)

with st.sidebar:
    sidebar.render()

if not session.has_credentials():
    st.info(
        "Add a provider API key in the sidebar to **generate** cases and variants, "
        "and to run a sweep. Without one you can still write cases and paste in a "
        "prompt you already have — the whole page stays usable."
    )

workbench.render()
