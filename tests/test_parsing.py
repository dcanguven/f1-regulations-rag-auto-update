"""Tests for PDF parsing and hierarchy extraction."""
from __future__ import annotations

import pytest

from app.parsing.pdf_parser import (
    HierarchyTracker,
    _classify_role,
    _extract_article_id,
    _is_header_footer,
)


# ── Article ID extraction ──────────────────────────────────────────────────────

@pytest.mark.parametrize("text, expected_id, has_title", [
    ("B5.13 Safety Car Procedure", "B5.13", True),
    ("B5.13.4", "B5.13.4", False),
    ("C14.3 Aerodynamic Testing", "C14.3", True),
    ("F1.1", "F1.1", False),
    ("This is not an article ID", None, False),
    ("The car must weigh at least 800kg", None, False),
])
def test_extract_article_id(text, expected_id, has_title):
    article_id, title = _extract_article_id(text)
    assert article_id == expected_id
    if has_title:
        assert title is not None
    if expected_id is None:
        assert article_id is None


# ── Header/footer detection ────────────────────────────────────────────────────

def test_header_detected_at_top():
    assert _is_header_footer("FIA Formula 1 Regulations", 1, 800, 10) is True


def test_footer_detected_at_bottom():
    assert _is_header_footer("Page 42", 1, 800, 770) is True


def test_body_text_not_filtered():
    assert _is_header_footer("B5.13 Safety Car Procedure", 1, 800, 400) is False


def test_issue_pattern_filtered():
    assert _is_header_footer("Issue 08 — 2026-08-05", 1, 800, 750) is True


# ── Content role classification ─────────────────────────────────────────────────

def test_black_text_is_regulatory():
    role = _classify_role("#000000", 0, 10.0)
    assert role == "regulatory_text"


def test_pink_is_regulatory_text():
    assert _classify_role("#ff00ff", 0, 10.0) == "regulatory_text"


def test_green_is_non_binding_comment():
    assert _classify_role("#00b050", 0, 10.0) == "non_regulatory_comment"


def test_red_is_governance_and_orange_is_reference():
    assert _classify_role("#c00000", 0, 10.0) == "governance_info"
    assert _classify_role("#ff9900", 0, 10.0) == "reference_info"


# ── Hierarchy tracker ──────────────────────────────────────────────────────────

def test_hierarchy_tracker_basic():
    tracker = HierarchyTracker("B")
    assert "B" in tracker.path()

    tracker.push("B5", "Race Procedure")
    assert tracker.current_article == "B5"
    assert "B5" in tracker.path()

    tracker.push("B5.13", "Safety Car")
    assert tracker.current_article == "B5.13"
    assert "B5.13" in tracker.path()


def test_hierarchy_tracker_path_with_sub():
    tracker = HierarchyTracker("C")
    tracker.push("C14", "Aero Testing")
    path = tracker.path_with_sub("C14.3")
    assert "C14.3" in path
