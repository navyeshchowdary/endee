from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

def get_embeddings(items):
    texts = [f"{item['title']} {item['desc']}" for item in items]
    return model.encode(texts)