"""Streamlit entry point — layout and wiring only.

Deliberately thin. Every decision this file could make has been pushed into
``core`` so it can be tested without a browser: what a confirmed brief is, when
a thread is stale, whether a metric configuration is usable, what a grade means.
What is left here is the shape of the page.

The five areas are tabs rather than pages because the workflow is a sequence a
user moves back and forth along — refining the brief after seeing a bad grade is
the normal case, not an exception.
"""

import streamlit as st

from prompt_workbench import __version__
from prompt_workbench.ui import candidates, define, evaluate, ground_truth, sidebar, test_area
from prompt_workbench.ui import session

st.set_page_config(page_title="Prompt Workbench", layout="wide")

workspace = session.workspace()

st.title("Prompt Workbench")
st.caption(f"v{__version__}")

st.markdown(
    "Turn a plain-language description of what you need a system prompt to do "
    "into artifacts you can actually test: a clarified brief, candidate system "
    "prompts written with different prompt techniques, a hybrid ground-truth "
    "dataset, manual model runs, and an evaluation you start yourself."
)

with st.sidebar:
    sidebar.render(workspace)

if not session.has_credentials():
    st.info(
        "Add a provider API key in the sidebar to enable generation, testing, "
        "and evaluation. Everything else stays editable without a key."
    )

define_tab, ground_truth_tab, candidates_tab, test_tab, evaluate_tab = st.tabs(
    ["Define", "Ground Truth", "Candidates", "Test", "Evaluate"]
)

with define_tab:
    define.render(workspace)
with ground_truth_tab:
    ground_truth.render(workspace)
with candidates_tab:
    candidates.render(workspace)
with test_tab:
    test_area.render(workspace)
with evaluate_tab:
    evaluate.render(workspace)
