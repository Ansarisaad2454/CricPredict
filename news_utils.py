# backend/news_utils.py
import os
from cachetools import cached, TTLCache
from typing import List, Dict, Any
from newsdataapi import NewsDataApiClient # <-- Import the new library

# Cache for 1 hour (3600 seconds)
news_cache = TTLCache(maxsize=1, ttl=3600)

@cached(cache=news_cache)
def fetch_cricket_news() -> List[Dict[str, Any]]:
    """
    Fetches cricket news from NewsData.io and caches the result for 1 hour.
    """
    print("--- FETCHING NEW NEWS FROM NewsData.io API ---")
    API_KEY = os.environ.get("NEWSDATA_API_KEY") # <-- Using a new env variable
    
    if not API_KEY:
        print("Error: NEWSDATA_API_KEY environment variable not set.")
        return []

    # Initialize the client
    api = NewsDataApiClient(apikey=API_KEY)
    
    try:
        # We search for "cricket" in the "sports" category, in English
        response = api.latest_api(q="cricket", category="sports", language="en")
        
        articles = response.get("results", [])
        
        print(f"--- API SUCCESS: Found {len(articles)} articles. ---")
        return articles

    except Exception as e:
        print(f"--- API FAILED: {e} ---") 
        return []