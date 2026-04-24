from endee import Endee, Precision

import os
ENDEE_URL = os.getenv("ENDEE_URL", "http://localhost:8080")
INDEX_NAME = "movies"
DIMS = 384

def get_client():
    client = Endee()
    client.set_base_url(f"{ENDEE_URL}/api/v1")
    return client

def store(items, embeddings):
    client = get_client()

    # Delete index if exists to avoid duplicates on restart
    try:
        client.delete_index(INDEX_NAME)
    except:
        pass

    client.create_index(
        name=INDEX_NAME,
        dimension=DIMS,
        space_type="cosine",
        precision=Precision.INT8
    )

    index = client.get_index(INDEX_NAME)

    vectors = [
        {
            "id": str(i),
            "vector": embeddings[i].tolist(),
            "meta": {
                "title": items[i]["title"],
                "genre": items[i]["genre"],
                "desc": items[i]["desc"]
            },
            "filter": {"genre": items[i]["genre"]}
        }
        for i in range(len(items))
    ]

    index.upsert(vectors)
    print(f"✅ Stored {len(items)} movies in Endee")