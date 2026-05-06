from dotenv import load_dotenv
import os
from vectordb import charger_index
from groq import Groq
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

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

