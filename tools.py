"""
tools.py

The three required FitFindr tools. Each tool is a standalone function that
can be called and tested independently before being wired into the agent loop.

Complete and test each tool before moving to agent.py.

Tools:
    search_listings(description, size, max_price)  → list[dict]
    suggest_outfit(new_item, wardrobe)              → str
    create_fit_card(outfit, new_item)               → str
"""

import os

from dotenv import load_dotenv
from groq import Groq

from utils.data_loader import load_listings

load_dotenv()


# ── Groq client ───────────────────────────────────────────────────────────────

def _get_groq_client():
    """Initialize and return a Groq client using GROQ_API_KEY from .env."""
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise ValueError(
            "GROQ_API_KEY not set. Add it to a .env file in the project root."
        )
    return Groq(api_key=api_key)


def _call_groq(system_prompt: str, user_prompt: str, temperature: float = 0.7) -> str:
    """Send a chat completion request to Groq and return the assistant reply."""
    client = _get_groq_client()
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
    )
    content = response.choices[0].message.content
    return content.strip() if content else ""


def _format_new_item(item: dict) -> str:
    """Format a listing dict into a readable block for LLM prompts."""
    colors = ", ".join(item.get("colors", [])) or "not specified"
    tags = ", ".join(item.get("style_tags", [])) or "none"
    brand = item.get("brand") or "unknown"
    return (
        f"- Title: {item.get('title', 'Unknown')}\n"
        f"- Description: {item.get('description', '')}\n"
        f"- Category: {item.get('category', '')}\n"
        f"- Size: {item.get('size', '')}\n"
        f"- Colors: {colors}\n"
        f"- Style tags: {tags}\n"
        f"- Brand: {brand}\n"
        f"- Price: ${item.get('price', '')} on {item.get('platform', '')}"
    )


def _format_wardrobe(items: list[dict]) -> str:
    """Format wardrobe items into a readable block for LLM prompts."""
    lines = []
    for piece in items:
        colors = ", ".join(piece.get("colors", []))
        tags = ", ".join(piece.get("style_tags", []))
        notes = piece.get("notes")
        notes_line = f"\n  Notes: {notes}" if notes else ""
        lines.append(
            f"- {piece.get('name', 'Unknown')} ({piece.get('category', '')})\n"
            f"  Colors: {colors}\n"
            f"  Style tags: {tags}{notes_line}"
        )
    return "\n".join(lines)


# ── Tool 1: search_listings ───────────────────────────────────────────────────

def search_listings(
    description: str,
    size: str | None = None,
    max_price: float | None = None,
) -> list[dict]:
    """
    Search the mock listings dataset for items matching the description,
    optional size, and optional price ceiling.

    Args:
        description: Keywords describing what the user is looking for
                     (e.g., "vintage graphic tee").
        size:        Size string to filter by, or None to skip size filtering.
                     Matching is case-insensitive (e.g., "M" matches "S/M").
        max_price:   Maximum price (inclusive), or None to skip price filtering.

    Returns:
        A list of matching listing dicts, sorted by relevance (best match first).
        Returns an empty list if nothing matches — does NOT raise an exception.

    Each listing dict has the following fields:
        id, title, description, category, style_tags (list), size,
        condition, price (float), colors (list), brand, platform

    TODO:
        1. Load all listings with load_listings().
        2. Filter by max_price and size (if provided).
        3. Score each remaining listing by keyword overlap with `description`.
        4. Drop any listings with a score of 0 (no relevant matches).
        5. Sort by score, highest first, and return the listing dicts.

    Before writing code, fill in the Tool 1 section of planning.md.
    """
    listings = load_listings()

    if max_price is not None:
        listings = [item for item in listings if item["price"] <= max_price]

    if size is not None:
        listings = [
            item for item in listings
            if size.lower() in item["size"].lower()
        ]

    keywords = [word.lower() for word in description.split() if word.strip()]

    def _score_listing(item: dict) -> int:
        parts = [
            item.get("title", ""),
            item.get("description", ""),
            item.get("category", ""),
            item.get("brand") or "",
            *item.get("style_tags", []),
            *item.get("colors", []),
        ]
        searchable = " ".join(parts).lower()
        return sum(1 for keyword in keywords if keyword in searchable)

    scored = []
    for item in listings:
        score = _score_listing(item)
        if score > 0:
            scored.append((item, score))
    scored.sort(key=lambda pair: pair[1], reverse=True)

    return [item for item, _ in scored]


# ── Tool 2: suggest_outfit ────────────────────────────────────────────────────

def suggest_outfit(new_item: dict, wardrobe: dict) -> str:
    """
    Given a thrifted item and the user's wardrobe, suggest 1–2 complete outfits.

    Args:
        new_item: A listing dict (the item the user is considering buying).
        wardrobe: A wardrobe dict with an 'items' key containing a list of
                  wardrobe item dicts. May be empty — handle this gracefully.

    Returns:
        A non-empty string with outfit suggestions.
        If the wardrobe is empty, offer general styling advice for the item
        rather than raising an exception or returning an empty string.

    TODO:
        1. Check whether wardrobe['items'] is empty.
        2. If empty: call the LLM with a prompt for general styling ideas
           (what kinds of items pair well, what vibe it suits, etc.).
        3. If not empty: format the wardrobe items into a prompt and ask
           the LLM to suggest specific outfit combinations using the new item
           and named pieces from the wardrobe.
        4. Return the LLM's response as a string.

    Before writing code, fill in the Tool 2 section of planning.md.
    """
    items = wardrobe.get("items") or []
    item_summary = _format_new_item(new_item)
    title = new_item.get("title", "this piece")

    system_prompt = (
        "You are a personal stylist helping someone style secondhand and thrifted finds. "
        "Give practical, specific outfit advice in a friendly, conversational tone."
    )

    if not items:
        user_prompt = f"""The user has an empty wardrobe — no saved clothing items yet.

They are considering buying this item:
{item_summary}

Suggest 1–2 general outfit ideas for this piece. Describe what kinds of items and colors pair well, the overall vibe or aesthetic, and any styling tips. Do not reference specific wardrobe pieces they own."""
    else:
        wardrobe_summary = _format_wardrobe(items)
        user_prompt = f"""The user is considering buying this new thrifted item:
{item_summary}

Their existing wardrobe:
{wardrobe_summary}

Suggest 1–2 complete outfits pairing the new item with specific pieces from their wardrobe (use each piece's name). Prioritize matches by shared style_tags, then colors, then category. Explain why each pairing works — mention item characteristics, colors, style vibes, and the overall aesthetic. Always include the new item's title ({title}) in your response.

IMPORTANT: If the new item's style (e.g. formal, black-tie, evening) is fundamentally incompatible with every piece in the wardrobe (e.g. all pieces are casual streetwear), do NOT invent a pairing. Instead respond with exactly the single token: STYLE_MISMATCH"""

    result = _call_groq(system_prompt, user_prompt)

    if not result:
        if not items:
            return (
                f"The {title} would pair well with versatile basics in a matching color palette. "
                "Try complementary bottoms and shoes that echo its style tags for a cohesive look."
            )
        return (
            f"Pair the {title} with wardrobe pieces that share its colors and style tags "
            "for a balanced, put-together outfit."
        )

    if "STYLE_MISMATCH" in result:
        return (
            f"The {title} has a style that doesn't pair well with the items currently in your wardrobe. "
            "Consider adding more pieces that match its aesthetic before building an outfit around it."
        )

    return result


# ── Tool 3: create_fit_card ───────────────────────────────────────────────────

def create_fit_card(outfit: str, new_item: dict) -> str:
    """
    Generate a short, shareable outfit caption for the thrifted find.

    Args:
        outfit:   The outfit suggestion string from suggest_outfit().
        new_item: The listing dict for the thrifted item.

    Returns:
        A 2–4 sentence string usable as an Instagram/TikTok caption.
        If outfit is empty or missing, return a descriptive error message
        string — do NOT raise an exception.

    The caption should:
    - Feel casual and authentic (like a real OOTD post, not a product description)
    - Mention the item name, price, and platform naturally (once each)
    - Capture the outfit vibe in specific terms
    - Sound different each time for different inputs (use higher LLM temperature)

    TODO:
        1. Guard against an empty or whitespace-only outfit string.
        2. Build a prompt that gives the LLM the item details and the outfit,
           and asks for a caption matching the style guidelines above.
        3. Call the LLM and return the response.

    Before writing code, fill in the Tool 3 section of planning.md.
    """
    if not outfit or not outfit.strip():
        return (
            "Error: could not generate a fit card caption. "
            "The outfit description is empty or incomplete — try running suggest_outfit() again."
        )

    item_summary = _format_new_item(new_item)
    title = new_item.get("title", "this piece")
    price = new_item.get("price", "")
    platform = new_item.get("platform", "")

    system_prompt = (
        "You write casual, authentic Instagram/TikTok OOTD captions for thrifted finds. "
        "Sound like a real person sharing their outfit — not a product listing. "
        "Keep it to 2–4 sentences. Vary your wording and style each time."
    )

    user_prompt = f"""You will write a shareable social media caption — but ONLY if the outfit suggestion below is valid.

STEP 1 — VALIDATE the outfit suggestion:
- Is it a real outfit description? (does it describe actual clothing items, combinations, or styling advice?)
- Does it mention or incorporate the thrifted item "{title}"?

If the outfit suggestion is NOT a real outfit description, OR does not incorporate "{title}", respond with ONLY the single token: INVALID_OUTFIT

STEP 2 — If valid, write a 2–4 sentence caption for this outfit.

New thrifted item:
{item_summary}

Outfit suggestion:
{outfit.strip()}

Caption requirements (only if valid):
- Mention "{title}" naturally (once)
- Mention the price (${price}) naturally (once)
- Mention the platform ({platform}) naturally (once)
- Capture the outfit vibe in specific terms
- Highlight the new item as the centerpiece
- Feel casual and authentic, like a real OOTD post"""

    result = _call_groq(system_prompt, user_prompt, temperature=0.95)

    if not result or "INVALID_OUTFIT" in result:
        return (
            "Error: could not generate a fit card caption. "
            "The outfit description is invalid or doesn't contain the new item — try running suggest_outfit() again."
        )

    return result
