from collections import Counter
import math

import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel, Field


DOCUMENTS = [
    "Machine learning models learn patterns from data.",
    "FastAPI is a Python framework for building APIs.",
    "Gradient descent optimizes model parameters.",
    "Computer vision uses images as input data.",
    "Retrieval systems rank documents against a search query.",
    "Python is widely used for machine learning and backend development.",
]


def tokenize(text):
    return [word.strip(".,!?;:()[]{}").lower() for word in text.split() if word.strip(".,!?;:()[]{}")]


class TfidfSearch:
    def __init__(self, documents):
        self.documents = documents
        tokenized = [tokenize(doc) for doc in documents]
        self.vocabulary = sorted(set(token for doc in tokenized for token in doc))
        index = {term: i for i, term in enumerate(self.vocabulary)}
        n = len(documents)

        matrix = np.zeros((n, len(self.vocabulary)), dtype=float)
        document_frequency = Counter()

        for row, doc in enumerate(tokenized):
            counts = Counter(doc)
            document_frequency.update(counts.keys())
            for term, count in counts.items():
                matrix[row, index[term]] = count / len(doc)

        idf = np.array([
            math.log((1 + n) / (1 + document_frequency[term])) + 1
            for term in self.vocabulary
        ])
        self.matrix = matrix * idf
        norms = np.linalg.norm(self.matrix, axis=1, keepdims=True)
        self.matrix = self.matrix / np.where(norms == 0, 1, norms)
        self.index = index
        self.idf = idf

    def vectorize(self, text):
        tokens = tokenize(text)
        counts = Counter(tokens)
        vector = np.zeros(len(self.vocabulary), dtype=float)
        total = max(len(tokens), 1)
        for term, count in counts.items():
            if term in self.index:
                vector[self.index[term]] = (count / total) * self.idf[self.index[term]]
        norm = np.linalg.norm(vector)
        return vector / norm if norm else vector

    def search(self, query, top_k=3):
        scores = self.matrix @ self.vectorize(query)
        order = np.argsort(scores)[::-1][:max(1, min(top_k, len(self.documents)))]
        return [
            {"document": self.documents[i], "score": round(float(scores[i]), 4)}
            for i in order
        ]


engine = TfidfSearch(DOCUMENTS)
app = FastAPI(title="Semantic Search Lab", version="0.1.0")


class SearchRequest(BaseModel):
    query: str
    top_k: int = Field(default=3, ge=1)


@app.get("/health")
def health():
    return {"status": "ok", "documents": len(DOCUMENTS)}


@app.post("/search")
def search(request: SearchRequest):
    return {"query": request.query, "results": engine.search(request.query, request.top_k)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
