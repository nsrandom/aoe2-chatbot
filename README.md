# AoE2 Chatbot

A simple self-hosted LLM that uses a RAG approach to answer questions about AoE2.

## Technologies Used

- Gemma3 LLM models
- Ollama to run the model
- Qdrant for RAG (vector db)

## Setup

#### 1. Run the Qdrant service via docker

```
docker compose up -d
```

#### 2. Set up the python environment

```
python -m venv .venv
source /.venv/bin/activate

pip install -r requirements
```

#### 3. Ingest some docs into the vector db

```
python fetch_page.py <url> rawdata/<filename>.md
```

#### 4. Run the chatbot

```
python chatbot.py
```
