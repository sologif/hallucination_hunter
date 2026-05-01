from duckduckgo_search import DDGS
import logging
import re

# Common English stopwords to exclude from keyword extraction
_STOPWORDS = {
    "the", "and", "that", "this", "with", "from", "have", "been", "were",
    "they", "their", "which", "about", "also", "some", "into", "more",
    "than", "when", "there", "what", "such", "even", "most", "each",
    "found", "study", "shows", "show", "find", "finds", "according",
    "suggests", "suggest", "reported", "reports", "published"
}

def _build_focused_query(text: str, max_tokens: int = 10) -> str:
    """
    Extract the most informative tokens from a claim to build a focused search query.
    Prioritizes: capitalized proper nouns, 4-digit years, and long content words.
    """
    tokens = text.split()
    selected = []

    for tok in tokens:
        clean = re.sub(r'[^\w]', '', tok)
        if not clean:
            continue

        # Always keep 4-digit years (e.g. 2021, 1969)
        if re.fullmatch(r'\d{4}', clean):
            selected.append(clean)
            continue

        # Keep capitalized words (proper nouns, institutions, etc.)
        if clean[0].isupper() and len(clean) > 1:
            selected.append(clean)
            continue

        # Keep long, meaningful lowercase words that aren't stopwords
        if len(clean) > 4 and clean.lower() not in _STOPWORDS:
            selected.append(clean)

    # Deduplicate while preserving order
    seen = set()
    unique = []
    for t in selected:
        if t.lower() not in seen:
            seen.add(t.lower())
            unique.append(t)

    focused = " ".join(unique[:max_tokens])
    # Fall back to a truncated version of the original if extraction yields nothing
    return focused if focused.strip() else text[:120].strip()


def search_web(query: str, limit: int = 5):
    """
    Search the web for sources using DuckDuckGo.
    Builds a focused, entity-aware query to maximise result relevance.
    """
    results = []

    # Build a tight, entity-aware query first
    focused_query = _build_focused_query(query)

    # Fallback chain: focused → full text (truncated) → even shorter
    queries_to_try = [focused_query]

    truncated = query.strip()[:150]
    if truncated.lower() != focused_query.lower():
        queries_to_try.append(truncated)

    if len(query) > 80:
        shorter = query.strip()[:80]
        if shorter not in queries_to_try:
            queries_to_try.append(shorter)

    try:
        with DDGS() as ddgs:
            for q in queries_to_try:
                search_results = list(ddgs.text(q, max_results=limit))
                if search_results:
                    for r in search_results:
                        results.append({
                            "title": r.get("title", ""),
                            "text": r.get("body", ""),
                            "url": r.get("href", ""),
                            "source": "web"
                        })
                    break  # Stop at the first query variant that returns results
    except Exception as e:
        logging.error(f"Web search error: {e}")

    return results


if __name__ == "__main__":
    # Test with the problematic coffee/IQ claim
    test_claim = "A 2021 Harvard Medical School study published in Nature found that drinking four cups of coffee daily increases IQ"
    print(f"Focused query: {_build_focused_query(test_claim)}\n")
    res = search_web(test_claim)
    for r in res:
        print(f"[{r['title']}]({r['url']})\n{r['text'][:120]}...\n")
