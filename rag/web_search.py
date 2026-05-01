from duckduckgo_search import DDGS
import logging

def search_web(query: str, limit: int = 3):
    """
    Search the web for sources using DuckDuckGo.
    Includes robustness fixes for casing and empty results.
    """
    results = []
    
    # 1. Normalize query for better DDG compatibility
    clean_query = query.strip()
    
    # Try searching with different variations if needed
    queries_to_try = [clean_query]
    
    # Fallback 1: Title case
    if not any(c.isupper() for c in clean_query):
        queries_to_try.append(clean_query.title())
    
    # Fallback 2: Uppercase short words (likely acronyms like USA, WWII, WW2)
    words = clean_query.split()
    caps_words = [w.upper() if len(w) <= 4 else w for w in words]
    caps_query = " ".join(caps_words)
    if caps_query.lower() != clean_query.lower() or caps_query != clean_query:
        if caps_query not in queries_to_try:
            queries_to_try.append(caps_query)
            
    # Fallback 3: Keyword-only (removes conversational filler)
    if len(words) > 3:
        keywords = " ".join([w for w in words if len(w) > 2])
        if keywords and keywords.lower() != clean_query.lower():
            if keywords not in queries_to_try:
                queries_to_try.append(keywords)

    try:
        with DDGS() as ddgs:
            for q in queries_to_try:
                # logging.info(f"Trying web search query: {q}")
                search_results = list(ddgs.text(q, max_results=limit))
                if search_results:
                    for r in search_results:
                        results.append({
                            "title": r.get("title", ""),
                            "text": r.get("body", ""),
                            "url": r.get("href", ""),
                            "source": "web"
                        })
                    break # Stop if we found results
    except Exception as e:
        logging.error(f"Web search error: {e}")
        
    return results

if __name__ == "__main__":
    # Test
    res = search_web("Who won the 2024 F1 championship?")
    for r in res:
        print(f"[{r['title']}]({r['url']})\n{r['text'][:100]}...\n")
