# 🎬 AI Movie Recommendation System

> A containerized, content-based recommendation engine using natural language processing and vector search — powered by **Endee Vector Database**.

---

## 📌 Overview

This project implements a semantic movie recommendation system that understands the *vibe* of what you're looking for — not just exact keywords. Ask for "mind-bending sci-fi" or "emotional love story" and it returns the most semantically similar movies from its database using vector embeddings and cosine similarity search.

Built as part of an AI/ML evaluation using the [Endee Vector Database](https://github.com/endee-io/endee).

---

## ✨ Features

- 🔍 **Semantic Search** — understands natural language queries, not just keywords
- 🎯 **Genre Filtering** — combines vector search with hard metadata filters via Endee
- 🧠 **Embedding-Based Recommendations** — uses `all-MiniLM-L6-v2` (384-dim vectors)
- 🐳 **Fully Dockerized** — single `docker compose up` starts everything
- 🌐 **Streamlit UI** — clean, interactive web interface at `localhost:8501`

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────┐
│                    User Browser                     │
│                  localhost:8501                     │
└─────────────────────┬───────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────┐
│              Streamlit Frontend (app.py)            │
│  - Accepts natural language query + genre filter    │
│  - Encodes query using SentenceTransformer          │
│  - Displays ranked recommendations                  │
└─────────────────────┬───────────────────────────────┘
                      │  vector query + filter
┌─────────────────────▼───────────────────────────────┐
│           Endee Vector Database (store/search)      │
│  - Stores 47 movie embeddings (384-dim, cosine)     │
│  - Performs top-K nearest neighbor search           │
│  - Supports metadata filtering by genre             │
│                  localhost:8080                     │
└─────────────────────────────────────────────────────┘
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Frontend UI | Streamlit |
| Embedding Model | `sentence-transformers` (`all-MiniLM-L6-v2`) |
| Vector Database | Endee (Docker) |
| Language | Python 3.10 |
| Infrastructure | Docker + Docker Compose |
| Vector Dimensions | 384 |
| Distance Metric | Cosine Similarity |
| Precision | INT8 |

---

## 📁 Project Structure

```
recommendation-project/
│
├── app.py              # Streamlit UI + main entry point
├── data.py             # Movie dataset (47 movies, 8 genres)
├── model.py            # SentenceTransformer embedding logic
├── store.py            # Endee index creation + vector upsert
├── search.py           # Endee vector search + genre filtering
├── requirements.txt    # Python dependencies
├── Dockerfile          # App container definition
├── docker-compose.yml  # Multi-container orchestration
└── README.md
```

---

## 🚀 Setup & Running

### Prerequisites
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed and running on your machine

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/recommendation-project.git
cd recommendation-project
```

### 2. Start everything with one command

```bash
docker compose up --build
```

This will:
- Pull and start the **Endee vector database** on port `8080`
- Build and start the **Streamlit app** on port `8501`
- Automatically load all movie embeddings into Endee on first run

### 3. Open the app

Visit **http://localhost:8501** in your browser.

### 4. Stop the app

```bash
docker compose down
```

---

## 💡 How It Works

1. **Data Preparation** — 47 movies across 8 genres, each with a title, genre tag, and descriptive text combining themes and keywords.

2. **Embedding Generation** — On startup, each movie's text is encoded into a 384-dimensional vector using `sentence-transformers/all-MiniLM-L6-v2`.

3. **Vector Storage** — Embeddings are upserted into an Endee index named `movies`, configured with cosine similarity and INT8 precision. Each vector carries metadata (`title`, `genre`, `desc`).

4. **Query Pipeline** — The user's natural language query is encoded using the same model. The resulting vector is sent to Endee's search API.

5. **Semantic Retrieval** — Endee returns the top-K most similar movies using approximate nearest neighbor (ANN) search with HNSW indexing.

6. **Hybrid Filtering** — If the user selects a genre, a metadata filter (`{"genre": {"$eq": "Action"}}`) is passed alongside the vector query — combining semantic similarity with hard logic filtering.

---

## 🎯 Example Queries

| Query | Top Result |
|---|---|
| `space adventure science` | Interstellar, The Martian, Gravity |
| `horror supernatural fear` | The Shining, Hereditary, It |
| `romantic emotional love` | The Notebook, Titanic, La La Land |
| `crime mafia underworld` | The Godfather, Goodfellas, The Departed |
| `magic fantasy wizard` | Harry Potter, Doctor Strange, The Hobbit |

---

## 🧩 Key Design Decisions

**Why Endee over a plain Python list?**
Endee provides persistent storage, HNSW-based ANN search, and metadata filtering — features that don't exist in a naive in-memory list. It also scales to millions of vectors without code changes.

**Why cosine similarity?**
Movie descriptions are encoded as normalized sentence embeddings. Cosine similarity measures the angle between vectors (semantic direction) rather than magnitude, making it ideal for text similarity tasks.

**Why `all-MiniLM-L6-v2`?**
It's fast, lightweight (90MB), runs on CPU, and produces high-quality 384-dim embeddings — a good balance for a dockerized demo without GPU requirements.

---

## 🔮 Future Improvements

- [ ] Add LLM-generated explanation for each recommendation ("Why this movie?")
- [ ] Expand dataset to 500+ movies using a public CSV
- [ ] Add user rating feedback loop to improve recommendations
- [ ] Deploy to cloud (Railway / Render / AWS EC2)
- [ ] Add RAG layer for natural language movie Q&A

---

## 👨‍💻 Author

**CH.Navyesh**  
Built for AI/ML evaluation — Endee Vector Database Project

---

## 📄 License

MIT License