from duckduckgo_search import DDGS
import re

def search_web(query: str, limit: int = 5):
    """
    Search the web for grounded evidence and filter out junk snippets.
    """
    results = []
    
    # Junk keywords to filter out irrelevant snippets (footers, navigation, ads)
    JUNK_KEYWORDS = [
        "cookie", "privacy policy", "all rights reserved", "subscribe now", 
        "sign up", "click here", "javascript", "browser", "login", "register",
        "join now", "warranty", "representation", "terms of service", "copyright"
    ]
    
    try:
        with DDGS() as ddgs:
            # We fetch more results than needed to filter out junk and keep high quality ones
            raw_results = list(ddgs.text(query, max_results=limit * 3))
            
            for r in raw_results:
                title = r.get("title", "")
                text = r.get("body", "")
                url = r.get("href", "")
                
                # Basic junk filtering
                if any(k in text.lower() or k in title.lower() for k in JUNK_KEYWORDS):
                    continue
                
                # Length check - very short or very long repetitive snippets are often junk
                if len(text) < 40 or len(text) > 1000:
                    continue
                
                # Relevance check: Does it contain any keywords from the query?
                # (Simple check to avoid completely unrelated results)
                query_words = set(re.findall(r'\w+', query.lower()))
                text_words = set(re.findall(r'\w+', text.lower()))
                overlap = len(query_words.intersection(text_words))
                
                if overlap > 0:
                    results.append({
                        "title": title,
                        "text": text,
                        "url": url,
                        "source": "web",
                        "relevance": overlap
                    })
            
            # Sort by relevance (keyword overlap) and take the top results
            results = sorted(results, key=lambda x: x["relevance"], reverse=True)[:limit]
            
            # Clean up the relevance key before returning
            for r in results:
                del r["relevance"]
                
    except Exception as e:
        print(f"Web search error: {e}")
        
    return results
