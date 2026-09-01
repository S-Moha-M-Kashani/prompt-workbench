"""Small pieces of presentation the workspace areas share."""

from __future__ import annotations

import streamlit as st

from prompt_workbench.core import context_limits
from prompt_workbench.models.provenance import SourceRef

# What a user is told when an action needs a key and there is none.
NO_KEY_HELP = "Add a provider API key in the sidebar to enable this."


def needs_credentials(has_key: bool) -> bool:
    """Show the setup notice and report whether the caller should stop."""
    if not has_key:
        st.info(NO_KEY_HELP)
    return not has_key


def needs_brief(confirmed: object | None) -> bool:
    """Every generated artifact starts from a confirmed brief; say so once."""
    if confirmed is None:
        st.info("Confirm a brief in the **Define** tab first — everything here is generated from it.")
        return True
    return False


def show_error(action: str, error: Exception) -> None:
    """Report a failed action without destroying anything the user has built."""
    st.error(f"{action} failed: {error}")


def provenance_caption(*refs: tuple[str, SourceRef | None]) -> None:
    """A one-line note saying which snapshots an artifact came from."""
    parts = [f"{label} {ref}" for label, ref in refs if ref is not None]
    if parts:
        st.caption("From " + " · ".join(parts))


def context_meter(used: int, budget: int) -> None:
    """How full the conversation's context window is, and when to worry."""
    fraction = context_limits.usage_fraction(used, budget)
    message = f"Context: about {used:,} of {budget:,} tokens ({fraction:.0%})"
    if fraction >= context_limits.WARN_AT:
        st.warning(f"{message} — clear the thread soon, nothing is trimmed automatically.")
    else:
        st.caption(message)
