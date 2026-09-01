"""The Streamlit surface, one module per workspace area.

Split out of ``app.py`` so each area stays readable and so ``app.py`` remains
what it claims to be: layout and wiring. Every module here may import from
``core``, ``services`` and ``models``; nothing in those layers may import from
here.
"""
