"""
agent.py

The FitFindr planning loop. Orchestrates the three tools in response to a
natural language user query, passing state between them via a session dict.

Complete tools.py and test each tool in isolation before implementing this file.

Usage (once implemented):
    from agent import run_agent
    from utils.data_loader import get_example_wardrobe

    result = run_agent(
        query="vintage graphic tee under $30, size M",
        wardrobe=get_example_wardrobe(),
    )
    print(result["fit_card"])
    print(result["error"])   # None on success
"""

import json
import re

from tools import search_listings, suggest_outfit, create_fit_card, _call_groq

MAX_SUGGEST_ATTEMPTS = 3


# ── session state ─────────────────────────────────────────────────────────────

def _new_session(query: str, wardrobe: dict) -> dict:
    """
    Initialize and return a fresh session dict for one user interaction.

    The session dict is the single source of truth for everything that happens
    during a run — it stores the original query, parsed parameters, tool results,
    and any error that caused early termination.

    You may add fields to this dict as needed for your implementation.
    """
    return {
        "query": query,              # original user query
        "parsed": {},                # extracted description / size / max_price
        "search_results": [],        # list of matching listing dicts
        "selected_item": None,       # top result, passed into suggest_outfit
        "wardrobe": wardrobe,        # user's wardrobe dict
        "outfit_suggestion": None,   # string returned by suggest_outfit
        "fit_card": None,            # string returned by create_fit_card
        "error": None,               # set if the interaction ended early
        "wardrobe_note": None,       # informational note (e.g. empty wardrobe)
        "close_matches": [],
        "suggest_attempts": 0,
    }


# ── query parsing ─────────────────────────────────────────────────────────────

_PRICE_PATTERNS = [
    re.compile(r"under\s+\$?\s*(\d+(?:\.\d+)?)", re.IGNORECASE),
    re.compile(r"below\s+\$?\s*(\d+(?:\.\d+)?)", re.IGNORECASE),
    re.compile(r"max(?:imum)?\s+(?:price\s+)?(?:of\s+)?\$?\s*(\d+(?:\.\d+)?)", re.IGNORECASE),
    re.compile(r"budget\s+of\s+\$?\s*(\d+(?:\.\d+)?)", re.IGNORECASE),
    re.compile(r"\$(\d+(?:\.\d+)?)\s*(?:or\s+less|max)?", re.IGNORECASE),
]

_SIZE_PATTERNS = [
    re.compile(
        r"(?:in\s+)?size\s+(\d+|[A-Za-z]+(?:/[A-Za-z]+)?)"
        r"(?=\s|,|$|under|below|max|budget|\$|\.)",
        re.IGNORECASE,
    ),
    re.compile(r"\bsize:\s*([A-Za-z0-9/]+)", re.IGNORECASE),
]

_STRIP_PATTERNS = [
    re.compile(r",?\s*under\s+\$?\s*\d+(?:\.\d+)?", re.IGNORECASE),
    re.compile(r",?\s*below\s+\$?\s*\d+(?:\.\d+)?", re.IGNORECASE),
    re.compile(r",?\s*max(?:imum)?\s+(?:price\s+)?(?:of\s+)?\$?\s*\d+(?:\.\d+)?", re.IGNORECASE),
    re.compile(r",?\s*budget\s+of\s+\$?\s*\d+(?:\.\d+)?", re.IGNORECASE),
    re.compile(r",?\s*\$\d+(?:\.\d+)?(?:\s*(?:or\s+less|max))?", re.IGNORECASE),
    re.compile(r",?\s*(?:in\s+)?size\s+[A-Za-z0-9/]+(?:\s+[A-Za-z0-9/]+)?", re.IGNORECASE),
    re.compile(r",?\s*size:\s*[A-Za-z0-9/]+", re.IGNORECASE),
]


def _parse_query_regex(query: str) -> dict:
    """Fallback parser: extract description, size, and max_price using regex."""
    text = query.strip()
    max_price = None
    size = None

    for pattern in _PRICE_PATTERNS:
        match = pattern.search(text)
        if match:
            max_price = float(match.group(1))
            break

    for pattern in _SIZE_PATTERNS:
        match = pattern.search(text)
        if match:
            size = match.group(1).strip()
            break

    description = text
    for pattern in _STRIP_PATTERNS:
        description = pattern.sub("", description)
    description = re.sub(r"\s{2,}", " ", description).strip(" ,.-")
    if not description:
        description = text.strip()

    return {
        "description": description,
        "size": size,
        "max_price": max_price,
    }


def _parse_query_llm(query: str) -> dict:
    """
    Use the LLM to extract description, size, and max_price from natural language.
    Returns a dict with those three keys, or raises on failure.
    """
    system_prompt = (
        "You are a query parser for a clothing search app. "
        "Extract structured fields from the user's natural language query. "
        "Respond with ONLY valid JSON — no markdown, no explanation. "
        'Format: {"description": "<item keywords only>", "size": "<size or null>", "max_price": <number or null>}'
    )
    user_prompt = (
        f"Query: {query}\n\n"
        "Rules:\n"
        "- description: concise item keywords only, no filler phrases like 'looking for' or 'I want'\n"
        "- size: exact size string if mentioned (e.g. 'S/M', 'M', 'L', '32'), else null\n"
        "- max_price: numeric budget if mentioned, else null"
    )
    raw = _call_groq(system_prompt, user_prompt, temperature=0)
    parsed = json.loads(raw)
    description = str(parsed.get("description") or "").strip()
    if not description:
        raise ValueError("LLM returned empty description")
    size_raw = parsed.get("size")
    size = str(size_raw).strip() if size_raw else None
    price_raw = parsed.get("max_price")
    max_price = float(price_raw) if price_raw is not None else None
    return {"description": description, "size": size, "max_price": max_price}


def _parse_query(query: str) -> dict:
    """Try LLM-based parsing first; fall back to regex on any failure."""
    try:
        return _parse_query_llm(query)
    except Exception:
        return _parse_query_regex(query)


# ── helpers ───────────────────────────────────────────────────────────────────


def _find_close_matches(
    description: str,
    size: str | None,
    max_price: float | None,
) -> list[dict]:
    """
    When strict search fails, find listings that match the description but
    fail the size and/or price constraints.
    """
    if size is None and max_price is None:
        return []

    desc_matches = search_listings(description, size=None, max_price=None)
    close = []
    for item in desc_matches:
        size_mismatch = (
            size is not None
            and size.lower() not in item.get("size", "").lower()
        )
        price_mismatch = (
            max_price is not None
            and item.get("price", float("inf")) > max_price
        )
        if size_mismatch or price_mismatch:
            close.append(item)
    return close[:5]


def _format_close_matches(
    close_matches: list[dict],
    size: str | None,
    max_price: float | None,
) -> str:
    lines = []
    for item in close_matches:
        mismatches = []
        if size is not None and size.lower() not in item.get("size", "").lower():
            mismatches.append(f"size is {item.get('size')} (you asked for {size})")
        if max_price is not None and item.get("price", 0) > max_price:
            mismatches.append(
                f"price is ${item.get('price'):.2f} (over your ${max_price:.2f} budget)"
            )
        reason = "; ".join(mismatches) if mismatches else "close match"
        lines.append(
            f"- {item.get('title')} — ${item.get('price'):.2f}, "
            f"{item.get('platform', 'unknown platform')}: {reason}"
        )
    return "\n".join(lines)


def _format_search_error(
    description: str,
    size: str | None,
    max_price: float | None,
    close_matches: list[dict],
) -> str:
    if close_matches:
        details = _format_close_matches(close_matches, size, max_price)
        return (
            "No listings matched all of your criteria, but I found similar items:\n"
            f"{details}\n\n"
            "Would you like to try one of these instead? "
            "Reply with an updated search (e.g. different size or budget) to continue."
        )
    return (
        f"No listings match the description \"{description}\". "
        "Try broadening your search terms or removing size/price filters."
    )


def _is_outfit_failure(outfit: str) -> bool:
    if not outfit or not outfit.strip():
        return True
    failure_markers = (
        "doesn't pair well",
        "does not pair well",
        "no suitable outfit",
        "STYLE_MISMATCH",
    )
    lowered = outfit.lower()
    return any(marker.lower() in lowered for marker in failure_markers)


def _new_item_in_outfit(outfit: str, new_item: dict) -> bool:
    title = (new_item.get("title") or "").strip()
    if not title:
        return False
    return title.lower() in outfit.lower()


def _is_fit_card_error(fit_card: str) -> bool:
    return bool(fit_card and fit_card.strip().startswith("Error:"))


# ── planning loop ─────────────────────────────────────────────────────────────

def run_agent(query: str, wardrobe: dict) -> dict:
    """
    Main agent entry point. Runs the FitFindr planning loop for a single
    user interaction and returns the completed session dict.

    Args:
        query:    Natural language user request
                  (e.g., "vintage graphic tee under $30, size M")
        wardrobe: User's wardrobe dict — use get_example_wardrobe() or
                  get_empty_wardrobe() from utils/data_loader.py

    Returns:
        The session dict after the interaction completes. Check session["error"]
        first — if it is not None, the interaction ended early and the other
        output fields (outfit_suggestion, fit_card) will be None.

    TODO — implement this function using the planning loop you designed in planning.md:

        Step 1: Initialize the session with _new_session().

        Step 2: Parse the user's query to extract a description, size, and
                max_price. You can use regex, string splitting, or ask the LLM
                to parse it — document your choice in planning.md.
                Store the result in session["parsed"].

        Step 3: Call search_listings() with the parsed parameters.
                Store results in session["search_results"].
                If no results: set session["error"] to a helpful message and
                return the session early. Do NOT proceed to suggest_outfit
                with empty input.

        Step 4: Select the item to use (e.g., the top result).
                Store it in session["selected_item"].

        Step 5: Call suggest_outfit() with the selected item and wardrobe.
                Store the result in session["outfit_suggestion"].

        Step 6: Call create_fit_card() with the outfit suggestion and selected item.
                Store the result in session["fit_card"].

        Step 7: Return the session.

    Before writing code, complete the Planning Loop and State Management sections
    of planning.md — your implementation should match what you described there.
    """
    session = _new_session(query, wardrobe)

    if not query or not query.strip():
        session["error"] = "Please enter what you're looking for."
        return session

    # Update session state
    parsed = _parse_query(query)
    session["parsed"] = parsed
    description = parsed["description"]
    size = parsed["size"]
    max_price = parsed["max_price"]

    # Search listings and handle failure modes
    search_results = search_listings(description, size=size, max_price=max_price)
    print(f"---> Called search_listings(description={description}, size={size}, max_price={max_price})")
    session["search_results"] = search_results

    # End session and asks for users' futher queries with close matches suggested
    if not search_results:
        close_matches = _find_close_matches(description, size, max_price)
        session["close_matches"] = close_matches
        session["error"] = _format_search_error(
            description, size, max_price, close_matches
        )
        return session

    # Search listings successful, moving on
    new_item = search_results[0]
    session["selected_item"] = new_item

    outfit_suggestion = None
    fit_card = None

    # Maximum of 3 attempts to suggest outfit and create fit card
    for attempt in range(1, MAX_SUGGEST_ATTEMPTS + 1):
        # Suggest a new outfit
        session["suggest_attempts"] = attempt
        outfit_suggestion = suggest_outfit(new_item, wardrobe)
        print(f"---> Called suggest_outfit(new_item={new_item}, wardrobe={wardrobe})\n")
        session["outfit_suggestion"] = outfit_suggestion

        # If cannot generate a suitable outfit
        if _is_outfit_failure(outfit_suggestion):
            title = new_item.get("title", "the new item")
            session["error"] = (
                f"Your wardrobe has items, but none pair well with {title}. "
                "Try a different listing or add pieces that match its style."
            )
            session["outfit_suggestion"] = outfit_suggestion
            return session

        wardrobe_empty = not wardrobe.get("items")

        # # Send out a note to user if wardrobe is empty -- NOT IMPLEMENTED
        # if wardrobe_empty:
        #     session["wardrobe_note"] = (
        #         "Your wardrobe is currently empty, so here's general styling advice "
        #         "for this item. Add pieces to your wardrobe to get personalized outfit combinations."
        #     )

        # Check if the new item is mentioned in the outfit
        if not wardrobe_empty and not _new_item_in_outfit(outfit_suggestion, new_item):
            if attempt >= MAX_SUGGEST_ATTEMPTS:
                session["error"] = (
                    "Internal error: could not build an outfit that includes the selected item. "
                    "Please try again."
                )
                return session
            continue

        # Create fit card
        fit_card = create_fit_card(outfit_suggestion, new_item)
        print(f"---> Called create_fit_card(outfit={outfit_suggestion}, new_item={new_item})\n")
        session["fit_card"] = fit_card

        # If there's an error on the fit card
        if _is_fit_card_error(fit_card):
            session["error"] = fit_card
            return session

        # Return the session if suggest_outfit() and create_fit_card() were successful
        return session
    
    # Otherwise, return the error
    session["error"] = (
        f"Internal error: outfit generation failed after {MAX_SUGGEST_ATTEMPTS} attempts. "
        "Please try again."
    )
    return session


# ── CLI test ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from utils.data_loader import get_example_wardrobe, get_empty_wardrobe

    print("=== Happy path: graphic tee ===\n")
    session = run_agent(
        query="looking for a vintage graphic tee under $30",
        wardrobe=get_example_wardrobe(),
    )
    if session["error"]:
        print(f"Error: {session['error']}")
    else:
        print(f"Found: {session['selected_item']['title']}")
        print(f"\nOutfit: {session['outfit_suggestion']}")
        print(f"\nFit card: {session['fit_card']}")

    print("\n\n=== No-results path ===\n")
    session2 = run_agent(
        query="designer ballgown size XXS under $5",
        wardrobe=get_example_wardrobe(),
    )
    print(f"Error message: {session2['error']}")


    scenarios = [
        ("Happy path: graphic tee", "looking for a vintage graphic tee under $30", get_example_wardrobe()),
        ("No-results path", "designer ballgown size XXS under $5", get_example_wardrobe()),
        ("Empty wardrobe", "vintage graphic tee under $30", get_empty_wardrobe()),
        ("Track jacket with budget", "90s track jacket in size M under $30", get_example_wardrobe()),
        ("Close match (strict size)", "graphic tee size XXS under $50", get_example_wardrobe()),
    ]

    for label, q, wardrobe in scenarios:
        print(f"\n{'=' * 60}\n=== {label} ===\nQuery: {q}")
        session = run_agent(query=q, wardrobe=wardrobe)
        print(f"Parsed: {session['parsed']}")
        if session["error"]:
            print(f"\nError: {session['error']}")
        else:
            print(f"\nFound: {session['selected_item']['title']}")
            if session["wardrobe_note"]:
                print(f"\nNote: {session['wardrobe_note']}")
            print(f"\nOutfit: {session['outfit_suggestion']}")
            print(f"\nFit card: {session['fit_card']}")
