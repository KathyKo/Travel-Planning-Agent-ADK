import os
import json
import re
from typing import List, Dict, Any

import numpy as np
import faiss
import requests
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

print("Tools (ADK): Loading local embedding model (all-MiniLM-L6-v2)...")
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
print("Tools (ADK): Local embedding model loaded.")

KB_DIR = "data"
KB_INDEX_PATH = os.path.join(KB_DIR, "knowledge_base.faiss")
KB_TEXTS_PATH = os.path.join(KB_DIR, "knowledge_base_texts.json")

kb_index = None
kb_texts: List[str] = []

print("Tools (ADK): Loading Knowledge Base from data/knowledge_base.faiss...")
if os.path.exists(KB_INDEX_PATH) and os.path.exists(KB_TEXTS_PATH):
    kb_index = faiss.read_index(KB_INDEX_PATH)
    with open(KB_TEXTS_PATH, "r", encoding="utf-8") as f:
        kb_texts = json.load(f)
    print("Tools (ADK): Knowledge Base loaded.")
else:
    print("Tools (ADK): Knowledge Base not found, starting with empty index.")
    dim = embedding_model.get_sentence_embedding_dimension()
    kb_index = faiss.IndexFlatL2(dim)
    kb_texts = []

PREFS_PATH = os.path.join(KB_DIR, "user_prefs.json")
print("Tools (ADK): Loading User Preferences DB from data/user_prefs.faiss...")
if not os.path.exists(PREFS_PATH):
    os.makedirs(KB_DIR, exist_ok=True)
    with open(PREFS_PATH, "w", encoding="utf-8") as f:
        json.dump({"preferences": []}, f, ensure_ascii=False, indent=2)
print("Tools (ADK): User Preferences DB loaded.")

def embed_text(text: str) -> np.ndarray:
    v = embedding_model.encode([text])
    return v.astype("float32")

def kb_search(query: str, top_k: int = 5) -> Dict[str, Any]:
    if kb_index is None or kb_index.ntotal == 0 or not kb_texts:
        return {
            "query": query,
            "results": [],
            "message": "Knowledge base is empty.",
        }
    q_vec = embed_text(query)
    distances, indices = kb_index.search(q_vec, top_k)
    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx < 0 or idx >= len(kb_texts):
            continue
        results.append(
            {
                "text": kb_texts[idx],
                "score": float(dist),
                "index": int(idx),
            }
        )
    return {
        "query": query,
        "results": results,
    }

def web_search(query: str, max_results: int = 5) -> Dict[str, Any]:
    load_dotenv()
    api_key = os.getenv("CUSTOM_SEARCH_API_KEY")
    cx = os.getenv("CUSTOM_SEARCH_CX")
    if not api_key or not cx:
        return {
            "query": query,
            "results": [],
            "error": "Missing CUSTOM_SEARCH_API_KEY or CUSTOM_SEARCH_CX in environment.",
            "source": "google_custom_search",
        }
    params = {
        "key": api_key,
        "cx": cx,
        "q": query,
        "num": max_results,
    }
    try:
        resp = requests.get(
            "https://www.googleapis.com/customsearch/v1",
            params=params,
            timeout=10,
        )
    except Exception as e:
        return {
            "query": query,
            "results": [],
            "error": f"Request error: {e}",
            "source": "google_custom_search",
        }
    if resp.status_code != 200:
        return {
            "query": query,
            "results": [],
            "error": f"HTTP {resp.status_code}: {resp.text}",
            "source": "google_custom_search",
        }
    data = resp.json()
    items = data.get("items", [])
    results = []
    for item in items[:max_results]:
        results.append(
            {
                "title": item.get("title", ""),
                "snippet": item.get("snippet", ""),
                "url": item.get("link", ""),
            }
        )
    return {
        "query": query,
        "results": results,
        "source": "google_custom_search",
    }

def search_flight_price(origin: str, destination: str, date: str, max_results: int = 10) -> Dict[str, Any]:
    load_dotenv()
    api_key = os.getenv("CUSTOM_SEARCH_API_KEY")
    cx = os.getenv("CUSTOM_SEARCH_CX")
    if not api_key or not cx:
        return {
            "origin": origin,
            "destination": destination,
            "date": date,
            "results": [],
            "error": "Missing CUSTOM_SEARCH_API_KEY or CUSTOM_SEARCH_CX in environment.",
            "source": "google_custom_search",
        }
    query = f"{origin} to {destination} flight {date} price"
    params = {
        "key": api_key,
        "cx": cx,
        "q": query,
        "num": max_results,
    }
    try:
        resp = requests.get(
            "https://www.googleapis.com/customsearch/v1",
            params=params,
            timeout=10,
        )
    except Exception as e:
        return {
            "origin": origin,
            "destination": destination,
            "date": date,
            "results": [],
            "error": f"Request error: {e}",
            "source": "google_custom_search",
        }
    if resp.status_code != 200:
        return {
            "origin": origin,
            "destination": destination,
            "date": date,
            "results": [],
            "error": f"HTTP {resp.status_code}: {resp.text}",
            "source": "google_custom_search",
        }
    data = resp.json()
    items = data.get("items", [])
    results = []
    for item in items[:max_results]:
        results.append(
            {
                "title": item.get("title", ""),
                "snippet": item.get("snippet", ""),
                "url": item.get("link", ""),
            }
        )
    return {
        "origin": origin,
        "destination": destination,
        "date": date,
        "results": results,
        "source": "google_custom_search",
    }

def get_weather(location: str, date: str = "today") -> Dict[str, Any]:
    return {
        "location": location,
        "date": date,
        "summary": "Partly cloudy with mild temperatures.",
        "temperature_celsius": 24.0,
        "precipitation_chance": 0.2,
        "source": "mock_weather_service",
    }

def save_preference(preference: str) -> Dict[str, Any]:
    try:
        with open(PREFS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        data = {"preferences": []}
    prefs = data.get("preferences", [])
    prefs.append(preference)
    data["preferences"] = prefs
    with open(PREFS_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return {
        "status": "ok",
        "saved_preference": preference,
        "total_preferences": len(prefs),
    }

def load_preferences() -> List[str]:
    try:
        with open(PREFS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        prefs = data.get("preferences", [])
        return prefs
    except Exception:
        return []

available_tools = [
    web_search,
    kb_search,
    get_weather,
    save_preference,
    load_preferences,
    search_flight_price,
]
