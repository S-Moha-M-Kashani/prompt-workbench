"""The brief: eight structured fields and the immutable snapshot of them."""

from datetime import UTC, datetime

import pytest

from prompt_workbench.models.brief import BRIEF_FIELDS, BriefSnapshot, PromptBrief


def test_a_new_brief_is_empty_and_lists_every_field_as_missing() -> None:
    brief = PromptBrief()
    assert brief.is_empty
    assert brief.missing_fields() == BRIEF_FIELDS
    assert brief.filled_fields() == ()


def test_brief_fields_cover_the_eight_discovery_topics() -> None:
    assert BRIEF_FIELDS == (
        "purpose",
        "audience",
        "inputs",
        "desired_behaviour",
        "constraints",
        "output_format",
        "examples",
        "failure_cases",
    )


def test_whitespace_only_text_does_not_count_as_filled() -> None:
    brief = PromptBrief(purpose="   \n  ")
    assert brief.is_empty
    assert "purpose" in brief.missing_fields()


def test_merging_keeps_existing_text_when_an_update_is_blank() -> None:
    brief = PromptBrief(purpose="Summarize support tickets", audience="agents")
    merged = brief.merged_with({"purpose": "", "audience": "team leads"})
    assert merged.purpose == "Summarize support tickets"
    assert merged.audience == "team leads"


def test_merging_rejects_a_field_the_brief_does_not_have() -> None:
    with pytest.raises(KeyError):
        PromptBrief().merged_with({"tone_of_voice": "friendly"})


def test_a_brief_is_immutable() -> None:
    brief = PromptBrief(purpose="x")
    with pytest.raises(Exception):
        brief.purpose = "y"  # type: ignore[misc]


def test_a_snapshot_records_its_identity_revision_and_time() -> None:
    when = datetime(2026, 3, 1, tzinfo=UTC)
    snapshot = BriefSnapshot(
        id="brief-1", revision=1, brief=PromptBrief(purpose="x"), created_at=when
    )
    assert (snapshot.id, snapshot.revision, snapshot.created_at) == ("brief-1", 1, when)


def test_a_snapshot_renders_only_its_filled_fields_as_context() -> None:
    snapshot = BriefSnapshot(
        id="brief-1",
        revision=1,
        brief=PromptBrief(purpose="Summarize tickets", constraints="Never invent facts"),
        created_at=datetime(2026, 3, 1, tzinfo=UTC),
    )
    text = snapshot.as_context()
    assert "Purpose: Summarize tickets" in text
    assert "Constraints: Never invent facts" in text
    assert "Audience" not in text


def test_two_snapshots_of_the_same_brief_compare_equal() -> None:
    when = datetime(2026, 3, 1, tzinfo=UTC)
    one = BriefSnapshot(id="b", revision=2, brief=PromptBrief(purpose="x"), created_at=when)
    two = BriefSnapshot(id="b", revision=2, brief=PromptBrief(purpose="x"), created_at=when)
    assert one == two
