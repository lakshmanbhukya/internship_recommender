"""
Engine selector based on environment
"""
import os

def get_search_engine():
    """Select search engine based on LIGHTWEIGHT_MODE env variable"""
    lightweight_mode = os.getenv("LIGHTWEIGHT_MODE", "false").lower() == "true"
    
    if lightweight_mode:
        print("[LIGHTWEIGHT] Using lightweight search engine (512 MB mode)")
        from api.lightweight_search import get_engine
        return get_engine()
    else:
        print("[FULL] Using hybrid search engine (full mode)")
        from api.hybrid_search import get_engine
        return get_engine()
