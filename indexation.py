import pandas as pd
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from vectordb import creer_index_faiss, sauvegarder_index

df = pd.read_csv("data/tmdb_5000_movies.csv")


def extraire_genres(genres_json):
    genres = json.loads(genres_json)
    return ", ".join([genre["name"] for genre in genres])

documents = []

for index, row in df.iterrows():
    titre = row["title"]
    synopsis = row["overview"]
    genres = extraire_genres(row["genres"])
    date_sortie = row["release_date"]
    note = row["vote_average"]
    langue = row["original_language"]
    duree = row["runtime"]

    contenu = f"""
Titre : {titre}
Genres : {genres}
Date de sortie : {date_sortie}
Note : {note}/10
Langue originale : {langue}
Durée : {duree} minutes
Synopsis : {synopsis}
"""

    documents.append({
        "id": f"film_{index}",
        "contenu": contenu,
        "metadata": {
            "titre": titre,
            "genres": genres,
            "date_sortie": date_sortie,
            "note": note,
            "langue": langue
        }
    })


def chunker(texte: str, taille_max: int = 500, overlap: int = 50) -> list[str]:
    paragraphs = texte.split("\n")
    chunks = []
    current_chunk = ""

    for p in paragraphs:
        if len(current_chunk) + len(p) < taille_max:
            current_chunk += p + "\n"
        else:
            chunks.append(current_chunk.strip())
            current_chunk = p + "\n"

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks
test = documents[0]["contenu"]
chunks = chunker(test)

for i, c in enumerate(chunks):
    print(f"\n--- Chunk {i} ---\n{c}")

chunks_avec_meta = []

for doc in documents:
    chunks = chunker(doc["contenu"])

    for chunk in chunks:
        chunks_avec_meta.append({
            "contenu": chunk,
            "metadata": doc["metadata"]
        })

print(f"Nombre de chunks : {len(chunks_avec_meta)}")
print(chunks_avec_meta[0])


modele = SentenceTransformer("all-mpnet-base-v2")

def embedder_chunks(chunks: list[str], modele) -> np.ndarray:
    vecteurs = modele.encode(
        chunks,
        convert_to_numpy=True,
        show_progress_bar=True
    )

    return vecteurs.astype("float32")

chunks_textes = [chunk["contenu"] for chunk in chunks_avec_meta]

vecteurs = embedder_chunks(chunks_textes, modele)

print(vecteurs.shape)

index = creer_index_faiss(vecteurs)
sauvegarder_index(index, chunks_avec_meta, "vector_store")

print("Index FAISS sauvegardé avec succès.")
print("Nombre de vecteurs indexés :", index.ntotal)