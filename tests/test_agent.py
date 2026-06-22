"""Tests for the FitFindr planning loop in agent.py."""

from unittest.mock import patch

import pytest

from agent import (
    MAX_SUGGEST_ATTEMPTS,
    _find_close_matches,
    _is_outfit_failure,
    _new_item_in_outfit,
    _parse_query,
    run_agent,
)
from tools import search_listings
from utils.data_loader import get_empty_wardrobe, get_example_wardrobe


# ── query parsing ─────────────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "query, expected_desc, expected_size, expected_price",
    [
        (
            "looking for a vintage graphic tee under $30",
            "looking for a vintage graphic tee",
            None,
            30.0,
        ),
        (
            "90s track jacket in size M",
            "90s track jacket",
            "M",
            None,
        ),
        (
            "vintage graphic tee under $30, size M",
            "vintage graphic tee",
            "M",
            30.0,
        ),
        (
            "black combat boots size 8 under $50",
            "black combat boots",
            "8",
            50.0,
        ),
    ],
)
def test_parse_query(query, expected_desc, expected_size, expected_price):
    parsed = _parse_query(query)
    assert parsed["description"] == expected_desc
    assert parsed["size"] == expected_size
    assert parsed["max_price"] == expected_price


# ── helper unit tests ─────────────────────────────────────────────────────────

def test_find_close_matches_finds_size_mismatch():
    close = _find_close_matches("vintage graphic tee", size="XXS", max_price=50.0)
    assert close
    assert all("XXS" not in item["size"] for item in close)


def test_new_item_in_outfit():
    item = {"title": "Graphic Tee — 2003 Tour Bootleg Style"}
    assert _new_item_in_outfit("Pair the Graphic Tee — 2003 Tour Bootleg Style with jeans.", item)
    assert not _new_item_in_outfit("Pair this tee with jeans.", item)


def test_is_outfit_failure():
    assert _is_outfit_failure("doesn't pair well with your wardrobe")
    assert not _is_outfit_failure("Pair the band tee with wide-leg jeans.")


# ── planning loop (mocked LLM) ────────────────────────────────────────────────

MOCK_OUTFIT = (
    "Pair the Graphic Tee — 2003 Tour Bootleg Style with your wide-leg jeans "
    "and platform Docs for a classic 90s grunge look."
)
MOCK_FIT_CARD = (
    "thrifted this Graphic Tee — 2003 Tour Bootleg Style off depop for $24 "
    "and honestly it was made for my wide-legs"
)


@patch("tools._call_groq")
def test_run_agent_happy_path(mock_groq):
    new_item = search_listings("vintage graphic tee", max_price=30)[0]
    title = new_item["title"]
    mock_outfit = (
        f"Pair the {title} with your wide-leg jeans "
        "and platform Docs for a classic 90s grunge look."
    )
    mock_groq.side_effect = [mock_outfit, MOCK_FIT_CARD]

    session = run_agent(
        query="vintage graphic tee under $30",
        wardrobe=get_example_wardrobe(),
    )

    assert session["error"] is None
    assert session["selected_item"] is not None
    assert session["outfit_suggestion"] == mock_outfit
    assert session["fit_card"] == MOCK_FIT_CARD
    assert session["parsed"]["max_price"] == 30.0


def test_run_agent_no_results():
    session = run_agent(
        query="designer ballgown size XXS under $5",
        wardrobe=get_example_wardrobe(),
    )

    assert session["error"] is not None
    assert session["selected_item"] is None
    assert session["outfit_suggestion"] is None
    assert session["fit_card"] is None


def test_run_agent_empty_query():
    session = run_agent(query="   ", wardrobe=get_example_wardrobe())
    assert session["error"] is not None


@patch("tools._call_groq")
def test_run_agent_empty_wardrobe(mock_groq):
    mock_groq.assert_not_called()

    session = run_agent(
        query="vintage graphic tee under $50",
        wardrobe=get_empty_wardrobe(),
    )

    assert session["error"] is not None
    assert "wardrobe is empty" in session["error"].lower()
    assert session["selected_item"] is not None
    assert session["outfit_suggestion"] is None
    mock_groq.assert_not_called()


@patch("tools._call_groq")
def test_run_agent_close_matches_in_error(mock_groq):
    session = run_agent(
        query="graphic tee size XXS under $50",
        wardrobe=get_example_wardrobe(),
    )

    if not session["search_results"]:
        assert session["error"] is not None
        assert session["close_matches"] or "No listings match" in session["error"]
    mock_groq.assert_not_called()


@patch("tools._call_groq")
def test_run_agent_style_mismatch(mock_groq):
    mock_groq.return_value = (
        "The Black Tie Evening Gown has a style that doesn't pair well "
        "with the items currently in your wardrobe."
    )

    formal_gown = {
        "id": "test-formal-001",
        "title": "Black Tie Evening Gown",
        "description": "Floor-length black-tie formal gown",
        "category": "dress",
        "style_tags": ["formal", "black-tie"],
        "size": "M",
        "price": 45.0,
        "colors": ["black"],
        "brand": "Unknown",
        "platform": "Depop",
    }

    with patch("agent.search_listings", return_value=[formal_gown]):
        session = run_agent(
            query="formal evening gown under $50",
            wardrobe=get_example_wardrobe(),
        )

    assert session["error"] is not None
    assert "pair well" in session["error"].lower() or "pair" in session["error"].lower()
    assert session["fit_card"] is None


@patch("tools._call_groq")
def test_run_agent_retries_when_item_missing_from_outfit(mock_groq):
    new_item = search_listings("vintage graphic tee", max_price=50)[0]
    title = new_item["title"]
    mock_groq.side_effect = [
        "Pair this tee with your jeans.",  # missing title — retry
        f"Pair {title} with your wide-leg jeans for a grunge look.",
        MOCK_FIT_CARD,
    ]

    session = run_agent(
        query="vintage graphic tee under $50",
        wardrobe=get_example_wardrobe(),
    )

    assert session["error"] is None
    assert session["suggest_attempts"] == 2
    assert title in session["outfit_suggestion"]
    assert mock_groq.call_count == 3


@patch("tools._call_groq")
def test_run_agent_internal_error_after_max_retries(mock_groq):
    mock_groq.return_value = "Pair this tee with your jeans."

    session = run_agent(
        query="vintage graphic tee under $50",
        wardrobe=get_example_wardrobe(),
    )

    assert session["error"] is not None
    assert "internal error" in session["error"].lower()
    assert session["suggest_attempts"] == MAX_SUGGEST_ATTEMPTS
