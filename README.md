# rag-movies-recommandation

# 🎬 Movie RAG Assistant

This project is a movie recommendation assistant built with a RAG (Retrieval-Augmented Generation) pipeline using Python, FAISS, Sentence Transformers, and Groq.

## Features

* Semantic search on movie data
* Movie recommendations using AI
* FAISS vector database
* Interactive command-line interface

## Technologies Used

* Python
* Pandas
* Sentence Transformers
* FAISS
* Groq API

## Project Structure

```id="8e3d8s"
├── indexation.py
├── rag.py
├── vectordb.py
├── contexte.txt
├── data/
├── vector_store/
└── README.md
```

## Installation

```bash id="hlrq11"
pip install -r requirements.txt
```

Create a `.env` file:

```id="9z3lhm"
GROQ_API_KEY=your_api_key
```

## Run the Project

### 1. Create the vector database

```bash id="9dvy8g"
python indexation.py
```

### 2. Start the RAG assistant

```bash id="vuwucv"
python rag.py
```

## Example

```id="u1zjqq"
Your question: I want a science fiction movie about AI
```

The assistant will search the database and generate a recommendation using the retrieved context.

## Dataset

TMDB 5000 Movie Dataset from Kaggle.

## Authors

Yakoub Kebaili / 
Melissa Djabella
