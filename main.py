from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import os
from pathlib import Path
import numpy as np
import faiss
import pickle
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import google.generativeai as genai

# ------------------------
# Config & Initialization
# ------------------------
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
g_model = genai.GenerativeModel("gemini-2.5-flash")

app = FastAPI()

# Static and Templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Files
CORPUS_FILE = "corpus.txt"
PROMPT_FILE = "prompt.txt"
FAISS_FILE = "faiss_index.bin"
ID_MAP_FILE = "id_to_chunk.pkl"

# Model for embeddings
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# Memory
index = None
id_to_chunk = {}
custom_prompt = ""

# ------------------------
# Utility Functions
# ------------------------
def chunk_text(text, chunk_size=200, overlap=50):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]))
        start += chunk_size - overlap
    return chunks

def get_embedding(text):
    return embed_model.encode(text, convert_to_numpy=True).astype(np.float32)

def load_index():
    global index, id_to_chunk
    if os.path.exists(FAISS_FILE) and os.path.exists(ID_MAP_FILE):
        index = faiss.read_index(FAISS_FILE)
        with open(ID_MAP_FILE, "rb") as f:
            id_to_chunk = pickle.load(f)
    else:
        index = None
        id_to_chunk = {}

def retrieve(query, top_k=3):
    if index is None or index.ntotal == 0:
        return ""
    query_emb = get_embedding(query).astype(np.float32)
    actual_top_k = min(top_k, index.ntotal)
    D, I = index.search(np.array([query_emb]), actual_top_k)
    return "\n\n".join([id_to_chunk[i] for i in I[0] if i != -1])

def rag_query(query, top_k=3):
    context = retrieve(query, top_k=top_k)
    prompt = f"""{custom_prompt}

Based on the following context, answer the user's question. 
If the context doesn't have enough info, say so.

Context:
{context}

User Question: {query}
"""
    try:
        response = g_model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"Error generating response: {e}"

# ------------------------
# Routes
# ------------------------
@app.get("/", response_class=HTMLResponse)
async def serve_home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/create_index")
async def create_index(corpus: str = Form(...)):
    global index, id_to_chunk
    # Save corpus
    with open(CORPUS_FILE, "w", encoding="utf-8") as f:
        f.write(corpus)

    # Chunk and embed
    chunks = chunk_text(corpus)
    if not chunks:
        return JSONResponse({"message": "Corpus is empty, no index created."})

    embeddings = [get_embedding(chunk) for chunk in chunks]
    dim = embeddings[0].shape[0]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings))

    id_to_chunk = {i: chunk for i, chunk in enumerate(chunks)}

    # Save index and chunks
    faiss.write_index(index, FAISS_FILE)
    with open(ID_MAP_FILE, "wb") as f:
        pickle.dump(id_to_chunk, f)

    return JSONResponse({"message": "Index created successfully!"})

@app.get("/get_corpus")
async def get_corpus():
    if os.path.exists(CORPUS_FILE):
        with open(CORPUS_FILE, "r", encoding="utf-8") as f:
            corpus = f.read()
    else:
        corpus = ""
    return {"corpus": corpus}

@app.post("/set_prompt")
async def set_prompt(prompt: str = Form(...)):
    global custom_prompt
    custom_prompt = prompt
    with open(PROMPT_FILE, "w", encoding="utf-8") as f:
        f.write(prompt)
    return {"message": "Prompt saved successfully!"}

@app.get("/get_prompt")
async def get_prompt():
    if os.path.exists(PROMPT_FILE):
        with open(PROMPT_FILE, "r", encoding="utf-8") as f:
            prompt = f.read()
    else:
        prompt = ""
    return {"prompt": prompt}

@app.post("/chat")
async def chat(query: str = Form(...)):
    answer = rag_query(query)
    return {"answer": answer}

# ------------------------
# On Startup
# ------------------------
@app.on_event("startup")
async def startup_event():
    global custom_prompt
    load_index()
    if os.path.exists(PROMPT_FILE):
        with open(PROMPT_FILE, "r", encoding="utf-8") as f:
            custom_prompt = f.read()
