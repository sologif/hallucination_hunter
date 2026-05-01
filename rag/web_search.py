from duckduckgo_search import DDGS

def search_web(query: str, limit: int = 5):
    results = []
    try:
        with DDGS() as ddgs:
            # results_gen = ddgs.text(query, max_results=limit)
            # ddgs.text returns a generator in newer versions, so we iterate
            for r in ddgs.text(query, max_results=limit):
                results.append({
                    "title": r.get("title", ""),
                    "text": r.get("body", ""),
                    "url": r.get("href", ""),
                    "source": "web"
                })
    except Exception as e:
        print(f"Web search error: {e}")
    return results
