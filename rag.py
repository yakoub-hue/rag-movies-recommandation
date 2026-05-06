from dotenv import load_dotenv
import os
from vectordb import charger_index
from groq import Groq
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
"""
# Test Groq
client = Groq(api_key=api_key)
response = client.chat.completions.create(
model="openai/gpt-oss-120b",
messages=[{"role": "user", "content": "Dis bonjour en une phrase."}]
)
print("■ Groq OK :", response.choices[0].message.content)

# Test Embeddings
model = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")
vector = model.encode("Test d'embedding")
print(f"■ Embedding OK — dimension : {len(vector)}")

# Test FAISS
index = faiss.IndexFlatL2(768)
index.add(np.array([vector], dtype=np.float32))
print(f"■ FAISS OK — {index.ntotal} vecteur(s) indexé(s)")
"""

def rechercher(question: str, modele, index, chunks_avec_meta: list, k: int = 4) -> list[dict]:

    question_vecteur = modele.encode([question]).astype("float32")

    distances, indices = index.search(question_vecteur, k)

    resultats = []

    for i in range(k):
        idx = indices[0][i]
        score = distances[0][i]

        chunk = chunks_avec_meta[idx]

        resultats.append({
            "contenu": chunk["contenu"],
            "metadata": chunk["metadata"],
            "score": float(score)
        })

    return resultats

modele = SentenceTransformer("all-mpnet-base-v2")

index, chunks_avec_meta = charger_index("vector_store")

resultats = rechercher(
    "film de science fiction avec intelligence artificielle",
    modele,
    index,
    chunks_avec_meta
)

for r in resultats:
    print("\n---")
    print(r["metadata"]["titre"])
    print("Score:", r["score"])