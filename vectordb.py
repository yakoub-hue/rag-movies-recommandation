import faiss
import json
import numpy as np
import os

def creer_index_faiss(vecteurs: np.ndarray) -> faiss.Index:
    vecteurs = vecteurs.astype("float32")
    dimension = vecteurs.shape[1]

    index = faiss.IndexFlatL2(dimension)
    index.add(vecteurs)

    return index


def sauvegarder_index(index, chunks_avec_meta: list, chemin: str):
    os.makedirs(chemin, exist_ok=True)

    faiss.write_index(index, f"{chemin}/index.faiss")

    with open(f"{chemin}/chunks.json", "w", encoding="utf-8") as f:
        json.dump(chunks_avec_meta, f, ensure_ascii=False, indent=2)


def charger_index(chemin: str):
    index = faiss.read_index(f"{chemin}/index.faiss")

    with open(f"{chemin}/chunks.json", "r", encoding="utf-8") as f:
        chunks_avec_meta = json.load(f)

    return index, chunks_avec_meta