from dotenv import load_dotenv
import os
from vectordb import charger_index
from groq import Groq
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

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


client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def construire_prompt_systeme() -> str:
    with open("contexte.txt", "r", encoding="utf-8") as f:
        return f.read()
def generer_reponse(question: str, chunks_pertinents: list[dict]) -> str:
    contexte = ""

    for i, chunk in enumerate(chunks_pertinents, start=1):
        metadata = chunk["metadata"]

        contexte += f"""
Source {i}
Titre : {metadata.get("titre")}
Note : {metadata.get("note")}/10
Genres : {metadata.get("genres")}
Langue : {metadata.get("langue")}
Date de sortie : {metadata.get("date_sortie")}

Contenu :
{chunk["contenu"]}
---
"""

    messages = [
        {
            "role": "system",
            "content": construire_prompt_systeme()
        },
        {
            "role": "user",
            "content": f"""
Question utilisateur :
{question}

Contexte récupéré depuis la base vectorielle :
{contexte}

Réponds à la question en utilisant uniquement ce contexte.
"""
        }
    ]

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=messages,
        temperature=0.3
    )

    return response.choices[0].message.content


def main():
    print("Chargement de la base de connaissances...")

    modele = SentenceTransformer("all-mpnet-base-v2")
    index, chunks_avec_meta = charger_index("vector_store")

    print("■ Système RAG prêt. Tapez 'quit' pour quitter.\n")

    while True:
        question = input("Votre question : ").strip()

        if question.lower() in ["quit", "exit", "q"]:
            print("Au revoir !")
            break

        if not question:
            continue

        chunks = rechercher(question, modele, index, chunks_avec_meta)

        reponse = generer_reponse(question, chunks)

        print("\n Réponse :\n")
        print(reponse)
        print("\n-----------------\n")

if __name__ == "__main__":
    main()