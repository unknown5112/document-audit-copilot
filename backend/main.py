import os
import io
import re
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from pypdf import PdfReader
import google.generativeai as genai
import chromadb
from chromadb.utils import embedding_functions

app = FastAPI(title="Cloud Document AI Copilot")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

# Ephemeral in-memory embeddings
emb_fn = embedding_functions.DefaultEmbeddingFunction()
chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection(
    name="cloud_documents",
    embedding_function=emb_fn
)

CURRENT_DIR = Path(__file__).resolve().parent
HTML_FILE_PATH = CURRENT_DIR / "index.html"

@app.get("/", response_class=HTMLResponse)
async def serve_root():
    if not HTML_FILE_PATH.exists():
        return HTMLResponse("<h3>index.html not found in backend directory</h3>", status_code=404)
    return HTMLResponse(content=HTML_FILE_PATH.read_text(encoding="utf-8"))

def clean_text(text: str) -> str:
    # Normalize whitespaces and linebreaks
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def chunk_text(text: str, chunk_size: int = 600, overlap: int = 150):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
    
    contents = await file.read()
    pdf_reader = PdfReader(io.BytesIO(contents))
    
    total_chunks = 0
    docs = []
    metas = []
    ids = []

    for page_idx, page in enumerate(pdf_reader.pages):
        raw_text = page.extract_text() or ""
        cleaned = clean_text(raw_text)
        if not cleaned:
            continue
        
        chunks = chunk_text(cleaned)
        for chunk_idx, chunk in enumerate(chunks):
            chunk_id = f"{file.filename}_p{page_idx+1}_c{chunk_idx}_{total_chunks}"
            docs.append(chunk)
            metas.append({
                "source": file.filename, 
                "page": page_idx + 1,
                "preview": chunk[:80] + "..."
            })
            ids.append(chunk_id)
            total_chunks += 1

    if not docs:
        raise HTTPException(
            status_code=400, 
            detail="No readable text extracted. If this is a scanned/image PDF, PyPDF cannot read text without OCR."
        )

    try:
        collection.upsert(
            documents=docs,
            metadatas=metas,
            ids=ids
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Indexing error: {str(e)}")

    return {"filename": file.filename, "status": "Indexed", "total_chunks": total_chunks}

class QueryRequest(BaseModel):
    query: str

@app.post("/query")
async def query_copilot(req: QueryRequest):
    if not GEMINI_API_KEY:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY environment variable is not set!")
        
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    
    try:
        # Check if collection is empty
        count = collection.count()
        if count == 0:
            return {
                "answer": "No documents are indexed in memory. Please upload a readable PDF file first.", 
                "sources": [],
                "retrieved_chunks": []
            }

        # Retrieve top-5 chunks for wider coverage
        n_fetch = min(5, count)
        results = collection.query(
            query_texts=[req.query],
            n_results=n_fetch
        )
        
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        
        if not docs:
            return {"answer": "No semantic match found for this query in the indexed PDF.", "sources": [], "retrieved_chunks": []}
        
        context_str = "\n\n---\n\n".join([f"[Page {m.get('page')}]: {d}" for d, m in zip(docs, metas)])
        
        prompt = f"""You are a helpful, enterprise-grade document auditor.
Analyze the provided excerpts from the uploaded document to answer the user's question directly and thoroughly.
Even if the exact phrase isn't present, synthesize closely related data, figures, or sections found in the excerpts.
Only if the provided text completely lacks any relevance, mention what topics are actually covered in the context.

Context Excerpts:
{context_str}

User Question: {req.query}
Direct Answer:"""

        # Auto-detect supported model
        valid_models = []
        for m in genai.list_models():
            if "generateContent" in m.supported_generation_methods:
                valid_models.append(m.name)

        sorted_models = sorted(valid_models, key=lambda x: (0 if "flash" in x else 1))
        
        response = None
        selected_model = None
        for m_name in sorted_models:
            try:
                model = genai.GenerativeModel(m_name)
                response = model.generate_content(prompt)
                if response and response.text:
                    selected_model = m_name
                    break
            except Exception:
                continue

        if not response:
            raise HTTPException(status_code=500, detail="Failed to generate answer from available models.")

        return {
            "answer": response.text, 
            "sources": metas,
            "retrieved_chunks": docs[:2],  # Send first 2 chunks for debugging
            "model_used": selected_model
        }

    except HTTPException:
        raise
    except Exception as e:
        print(f"[Query Error]: {e}")
        raise HTTPException(status_code=500, detail=f"Generation error: {str(e)}")