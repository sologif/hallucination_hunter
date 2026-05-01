from duckduckgo_search import DDGS
import logging

def search_web(query: str, limit: int = 3):
    """
    Search the web for sources using DuckDuckGo.
    """
    results = []
    try:
        with DDGS() as ddgs:
            # text search
            search_results = list(ddgs.text(query, max_results=limit))
            for r in search_results:
                results.append({
                    "title": r.get("title", ""),
                    "text": r.get("body", ""),
                    "url": r.get("href", ""),
                    "source": "web"
                })
    except Exception as e:
        logging.error(f"Web search error: {e}")
        
    return results

if __name__ == "__main__":
    # Test
    res = search_web("Who won the 2024 F1 championship?")
    for r in res:
        print(f"[{r['title']}]({r['url']})\n{r['text'][:100]}...\n")
