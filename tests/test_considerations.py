"""The limits that must stay next to the numbers, from one source."""

from pathlib import Path

import pytest

from prompt_workbench.core import considerations

DOC = Path(__file__).resolve().parents[1] / "docs" / "LIMITATIONS.md"


def test_the_required_statements_are_all_present() -> None:
    keys = {item.key for item in considerations.all_considerations()}
    assert keys >= {
        "single_run_latency",
        "judged_grades",
        "mocked_tools",
        "one_round",
        "unknown_cost",
    }


@pytest.mark.parametrize(
    "item", considerations.all_considerations(), ids=lambda i: i.key
)
def test_every_statement_is_written_out_and_placed(item) -> None:
    assert item.title.strip()
    assert item.statement.strip()
    assert item.shown_beside, f"{item.key} names no number it qualifies"


def test_statements_can_be_selected_by_what_is_on_screen() -> None:
    beside_latency = {i.key for i in considerations.beside("latency")}
    assert "single_run_latency" in beside_latency
    assert "judged_grades" not in beside_latency


def test_the_document_and_the_page_share_one_source() -> None:
    """Two copies of "a single run is not a benchmark" disagree within a
    release, and the one that disagrees is the one next to the number.

    Compared with whitespace normalised: the document is hard-wrapped and the
    screen is not, so what must match is the sentences, not the line breaks.
    """
    text = " ".join(DOC.read_text().split())
    for item in considerations.all_considerations():
        assert " ".join(item.statement.split()) in text, item.key


def test_the_document_names_each_statement_as_its_own_heading() -> None:
    text = DOC.read_text()
    for item in considerations.all_considerations():
        assert f"### {item.title}" in text, item.key
