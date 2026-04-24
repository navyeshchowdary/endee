from endee import Endee

import os
ENDEE_URL = os.getenv("ENDEE_URL", "http://localhost:8080")
INDEX_NAME = "movies"

def get_index():
    client = Endee()
    client.set_base_url(f"{ENDEE_URL}/api/v1")
    return client.get_index(INDEX_NAME)

def search(query_embedding, top_k=5, genre_filter=None):
    index = get_index()

    params = {
        "vector": query_embedding.tolist(),
        "top_k": top_k,
        "ef": 128,
        "include_vectors": False
    }

    if genre_filter and genre_filter != "All":
        params["filter"] = [{"genre": {"$eq": genre_filter}}]

    results = index.query(**params)
    return [
        {
            "title": r["meta"]["title"],
            "genre": r["meta"]["genre"],
            "score": round(r["similarity"], 4)
        }
        for r in results
    ]