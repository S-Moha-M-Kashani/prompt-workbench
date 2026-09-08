"""Framework adapters: the one ring allowed to import a framework SDK.

A sibling of ``services/`` rather than a member of it. ``services/`` is "one
module per external system"; frameworks are a different axis — many ways to
reach the *same* system. ``core/`` depends only on the ``CallRunner`` protocol
in ``models/``, so adding a framework is a new module plus a registry entry and
touches neither the call shape, the measurement, nor the UI.

Consequence, stated rather than hidden: ``services/openrouter_client`` is no
longer the only module that opens a connection. The rule is now narrower — it is
the only module that speaks the provider's own API directly, and ``llm_call/``
is the only other place a connection is opened, always through a framework.
"""
