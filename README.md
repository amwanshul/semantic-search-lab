# Semantic Search Lab

A compact retrieval system built to understand the mechanics behind search and RAG pipelines.

## Pipeline

```
documents → tokenization → TF-IDF → cosine similarity → ranked results → API
```

## What it demonstrates

- Inverted vocabulary construction
- Term frequency / inverse document frequency weighting
- Vector normalization
- Cosine similarity
- Top-k ranking
- A minimal FastAPI interface

This project deliberately uses classical information retrieval instead of an embedding API. That makes the retrieval behaviour inspectable and gives a useful baseline for later experiments with dense embeddings.

## Run

```bash
pip install -r requirements.txt
python app.py
```

Then open the FastAPI docs at `/docs`.

## Example

POST `/search` with:

```json
{"query":"fast api machine learning","top_k":3}
```

## Scope

This is an educational retrieval engine, not a production search service. It is designed to make ranking behaviour easy to inspect and extend.
