# tests/test_tools.py
# import sys
# from pathlib import Path

# sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from unittest.mock import patch

from tools import search_listings, suggest_outfit, create_fit_card
from utils.data_loader import get_example_wardrobe, get_empty_wardrobe

def test_search_returns_results():
    results = search_listings("vintage graphic tee", size=None, max_price=50)
    # print(f"test_search_returns_results() = {repr(results)}")
    assert isinstance(results, list)
    assert len(results) > 0

def test_search_empty_results():
    results = search_listings("designer ballgown", size="XXS", max_price=5)
    # print(f"test_search_empty_results() = {repr(results)}")
    assert results == []   # empty list, no exception

def test_search_price_filter():
    results = search_listings("jacket", size=None, max_price=10)
    # print(f"search_price_filter() = {repr(results)}")
    assert all(item["price"] <= 10 for item in results)

def test_suggest_outfit_wardrobe_empty():
    new_item = search_listings("vintage graphic tee", size=None, max_price=50)[0]
    print(f"new_item = {repr(new_item)}")
    results = suggest_outfit(new_item, wardrobe=get_empty_wardrobe())
    print(f"results = {repr(results)}")
    assert results and len(results) > 0

def test_suggest_outfit_unsuitable_style():
    # Claude Code generated test
    """LLM reports it cannot build an outfit due to a style clash."""
    formal_gown = {
        "id": "test-formal-001",
        "title": "Black Tie Evening Gown",
        "description": "Floor-length black-tie formal gown with sequins",
        "category": "dress",
        "style_tags": ["formal", "black-tie", "evening"],
        "size": "M",
        "condition": "like new",
        "price": 45.00,
        "colors": ["black"],
        "brand": "Unknown",
        "platform": "Depop",
    }
    streetwear_wardrobe = {
        "items": [
            {
                "name": "Oversized Hoodie",
                "category": "top",
                "colors": ["grey"],
                "style_tags": ["streetwear", "casual", "oversized"],
            },
            {
                "name": "Cargo Pants",
                "category": "bottom",
                "colors": ["olive"],
                "style_tags": ["streetwear", "utility", "casual"],
            },
        ]
    }
    # style_clash_message = (
    #     "I'm sorry, but the Black Tie Evening Gown's formal aesthetic is incompatible "
    #     "with the casual streetwear pieces in your wardrobe. No suitable outfit could be formed."
    # )

    # with patch("tools._call_groq", return_value=style_clash_message):
    result = suggest_outfit(formal_gown, streetwear_wardrobe)

    print(f"result = {repr(result)}")
    assert result                                      # not empty
    assert "STYLE_MISMATCH" not in result  
    assert "Black Tie Evening Gown" in result          # item is mentioned

def test_create_fit_card_improper_outfit():
    random_string = "This is DEFINITELY not an outfit."
    empty_string = ""
    whitespaces = "           "
    null = None
    new_item = search_listings("vintage graphic tee", size=None, max_price=50)[0]
    res_random_string = create_fit_card(outfit=random_string, new_item=new_item)
    res_empty_string = create_fit_card(outfit=empty_string, new_item=new_item)
    res_whitespaces = create_fit_card(outfit=whitespaces, new_item=new_item)
    res_null = create_fit_card(outfit=null, new_item=new_item)
    print(f"res_random_string = {repr(res_random_string)}")
    print(f"res_empty_string = {repr(res_empty_string)}")
    print(f"res_whitespaces = {repr(res_whitespaces)}")
    print(f"res_null = {repr(res_null)}")
    EMPTY_ERROR = (
        "Error: could not generate a fit card caption. "
        "The outfit description is empty or incomplete — try running suggest_outfit() again."
    )
    INVALID_ERROR = (
        "Error: could not generate a fit card caption. "
        "The outfit description is invalid or doesn't contain the new item — try running suggest_outfit() again."
    )
    assert res_empty_string == EMPTY_ERROR
    assert res_whitespaces == EMPTY_ERROR
    assert res_null == EMPTY_ERROR
    assert res_random_string == INVALID_ERROR


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))