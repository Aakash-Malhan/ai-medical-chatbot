import os
import time
import tempfile
from typing import List, Tuple

# Keep model caches small & ephemeral on HF Spaces
os.environ.setdefault("HF_HOME", "/tmp/.cache")
os.environ.setdefault("HF_HUB_CACHE", "/tmp/.cache/huggingface")
os.environ.setdefault("FASTEMBED_CACHE_PATH", "/tmp/.cache/fastembed")

import gradio as gr
from dotenv import load_dotenv
from pypdf import PdfReader

# ---- Embeddings (FastEmbed) ----
from fastembed import TextEmbedding
import numpy as np

# ---- Text splitting (new/old LC) ----
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except Exception:  # LC < 0.2
    from langchain.text_splitter import RecursiveCharacterTextSplitter  # type: ignore

# ---- Pinecone ----
try:
    from pinecone import Pinecone, ServerlessSpec
    PINECONE_OK = True
except Exception:
    PINECONE_OK = False

load_dotenv()

# ----------------- Env -----------------
MODEL_PROVIDER = os.getenv("MODEL_PROVIDER", "GEMINI").upper()
MODEL_NAME = os.getenv("MODEL_NAME", "gemini-2.0-flash")  # 2.0 Flash by default

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "medical-rag-idx")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT", "us-east-1")

# MiniLM (FastEmbed ONNX). 384-dim cosine.
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

DISCLAIMER = (
    "This chatbot is designed for informational purposes only and should not be used as a "
    "substitute for professional medical advice, diagnosis, or treatment. Always seek the "
    "advice of your physician or other qualified health provider with any questions you may "
    "have regarding a medical condition."
)

# ----------------- Gemini LLM (2.0-first, probe-first selector) -----------------
_LLM_GENERATE = None
def _ensure_llm():
    """Return generate(prompt:str)->str; tries 2.0 Flash first, then robust fallbacks."""
    global _LLM_GENERATE
    if _LLM_GENERATE is not None:
        return _LLM_GENERATE

    if MODEL_PROVIDER != "GEMINI":
        raise RuntimeError(f"Unsupported MODEL_PROVIDER: {MODEL_PROVIDER}")

    import google.generativeai as genai
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing GEMINI_API_KEY")
    genai.configure(api_key=api_key)

    preferred = os.getenv("MODEL_NAME", "gemini-2.0-flash")

    candidates = [
        # 2.0 Flash family
        preferred,
        "gemini-2.0-flash",
        "gemini-2.0-flash-latest",
        "gemini-2.0-flash-exp",   # some accounts
        # Common 1.5 fallbacks (widely available)
        "gemini-1.5-flash-8b",
        "gemini-1.5-flash-latest",
        "gemini-1.5-flash",
        "gemini-1.5-pro-latest",
        "gemini-1.5-pro",
        "gemini-1.0-pro",
    ]

    # Deduplicate while preserving order
    seen = set()
    candidates = [c for c in candidates if c and not (c in seen or seen.add(c))]

    chosen = None
    last_err = None
    for mid in candidates:
        try:
            m = genai.GenerativeModel(mid)
            # tiny probe to verify permission & route
            _ = m.generate_content("hi").text
            chosen = m
            break
        except Exception as e:
            last_err = e
            continue

    if chosen is None:
        try:
            avail = []
            for md in genai.list_models():
                n = getattr(md, "name", "")
                methods = set(getattr(md, "supported_generation_methods", []) or [])
                if n and "generateContent" in methods:
                    avail.append(n)
            print("Gemini list_models that support generateContent:", avail)
        except Exception as _e:
            print("list_models failed:", _e)
        raise RuntimeError(f"No Gemini model supporting generateContent is available to your key. Last error: {last_err}")

    def generate(prompt: str) -> str:
        try:
            out = chosen.generate_content(prompt)
            return (out.text or "").strip()
        except Exception:
            # last-resort backup to a broadly available small model
            backup = genai.GenerativeModel("gemini-1.5-flash-8b")
            return (backup.generate_content(prompt).text or "").strip()

    _LLM_GENERATE = generate
    return _LLM_GENERATE

# ----------------- Embeddings helpers -----------------
_embedder = None
def init_embeddings():
    global _embedder
    if _embedder is None:
        _embedder = TextEmbedding(model_name=EMBED_MODEL_NAME)  # downloads ONNX to /tmp

def encode_texts(texts: List[str]) -> List[List[float]]:
    init_embeddings()
    vecs = list(_embedder.embed(texts))  # generator of np arrays
    if not vecs:
        return []
    arr = np.vstack(vecs).astype(np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12
    arr = arr / norms  # L2-normalize for cosine
    return arr.tolist()

# ----------------- Pinecone -----------------
pc = None
index = None
def init_pinecone():
    global pc, index
    if not PINECONE_OK or not PINECONE_API_KEY:
        return
    if pc is None:
        pc = Pinecone(api_key=PINECONE_API_KEY)
    existing = [i["name"] for i in pc.list_indexes()]
    if PINECONE_INDEX_NAME not in existing:
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=384,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region=PINECONE_ENVIRONMENT),
        )
        time.sleep(5)  # allow serverless index to be ready
    index = pc.Index(PINECONE_INDEX_NAME)

def pinecone_upsert(vectors: List[Tuple[str, List[float], dict]]):
    if index is None:
        raise RuntimeError("Pinecone is not initialized or API key missing.")
    index.upsert([{"id": _id, "values": vec, "metadata": meta} for _id, vec, meta in vectors])

def pinecone_query(query_vec: List[float], top_k=5) -> List[dict]:
    if index is None:
        return []
    res = index.query(vector=query_vec, top_k=top_k, include_metadata=True)
    out = []
    for m in res.matches or []:
        meta = m.metadata if hasattr(m, "metadata") else (m.get("metadata", {}) if isinstance(m, dict) else {})
        score = float(m.score if hasattr(m, "score") else m.get("score", 0.0))
        out.append({"text": meta.get("text", ""), "source": meta.get("source", ""), "score": score})
    return out

# ----------------- PDF → chunks -----------------
def read_pdfs(files) -> List[Tuple[str, str]]:
    docs = []
    for f in files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(f.read()); tmp.flush()
            reader = PdfReader(tmp.name)
            pages = []
            for p in reader.pages:
                try:
                    pages.append(p.extract_text() or "")
                except Exception:
                    pages.append("")
            docs.append((f.name, "\n".join(pages).strip()))
    return docs

def chunk_text(source: str, text: str, chunk_size=900, chunk_overlap=120) -> List[Tuple[str, str, str]]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, separators=["\n\n", "\n", " ", ""]
    )
    chunks = splitter.split_text(text)
    return [(f"{source}-{i}", c, source) for i, c in enumerate(chunks)]

def embed_and_store(chunks: List[Tuple[str, str, str]]) -> int:
    if not chunks:
        return 0
    init_embeddings(); init_pinecone()
    ids, texts, sources = zip(*chunks)
    vectors = encode_texts(list(texts))
    pinecone_upsert([(ids[i], vectors[i], {"text": texts[i], "source": sources[i]}) for i in range(len(texts))])
    return len(texts)

# ----------------- Answering -----------------
def build_rag_prompt(question: str, contexts: List[dict]) -> str:
    ctx = "\n\n".join([f"[Source: {c['source']}]\n{c['text'][:1600]}" for c in contexts])
    return f"""
You are a careful medical information assistant.
Use ONLY the context below to answer the user's question. If the answer is not present in the context, say you don't know.
Context:
{ctx}
Question: {question}
Guidelines:
- Keep the answer concise and clear (5–10 sentences).
- Do not diagnose or prescribe treatment.
- Mention uncertainties and suggest consulting a clinician when appropriate.
- End with up to three bracketed citations like [source.pdf].
Answer:
""".strip()

def general_answer(question: str) -> str:
    llm = _ensure_llm()
    prompt = f"""
You are a careful medical information assistant.
The user asked: {question}
Provide a brief, general-information overview (5–8 sentences) that may include:
- What it is (simple definition)
- Common symptoms
- General risk factors/causes
- Broad, non-prescriptive precautions and when to seek urgent care
Avoid diagnosis, dosing, or tailored treatment plans. Keep it factual and cautious.
Finish the answer before the disclaimer; do NOT include citations.
Answer:
""".strip()
    return llm(prompt).strip()

def retrieve_and_answer(question: str, top_k=5):
    used_rag, srcs = False, []
    try:
        init_embeddings(); init_pinecone()
        if index is not None:
            qvec = encode_texts([question])[0]
            hits = pinecone_query(qvec, top_k=top_k)
            if hits and any(h["score"] >= 0.2 for h in hits):
                llm = _ensure_llm()
                ans = llm(build_rag_prompt(question, hits)).strip()
                for h in hits[:3]:
                    if h["source"] not in srcs:
                        srcs.append(h["source"])
                used_rag = True
                return ans, srcs, used_rag
    except Exception:
        pass
    return general_answer(question), srcs, used_rag

# ----------------- Gradio UI -----------------
with gr.Blocks(title="AI Medical ChatBot (RAG + Fallback)") as demo:
    gr.Markdown(
        "## 🩺 AI Medical ChatBot\n"
        "Upload PDFs to get **document-grounded** answers with sources, or ask general medical questions "
        "for a **non-diagnostic** fallback answer.\n"
    )

    with gr.Tab("Upload & Index"):
        uploader = gr.File(label="Upload medical PDFs", file_count="multiple", file_types=[".pdf"])
        idx_btn = gr.Button("Create / Update Knowledge Index")
        idx_status = gr.Markdown()

        def do_index(files):
            if not files:
                return "Please upload at least one PDF."
            if not PINECONE_OK or not PINECONE_API_KEY:
                return "❗ Pinecone is not configured. Set PINECONE_API_KEY to enable document indexing."
            docs = read_pdfs(files); total = 0
            for src, txt in docs:
                if txt.strip():
                    total += embed_and_store(chunk_text(src, txt))
            return f"✅ Indexed {len(docs)} documents with {total} chunks into `{PINECONE_INDEX_NAME}`."

        idx_btn.click(fn=do_index, inputs=uploader, outputs=idx_status)

    with gr.Tab("Ask"):
        question = gr.Textbox(
            label="Your question",
            placeholder="e.g., What are the common symptoms of influenza and what general precautions help?"
        )
        ask_btn = gr.Button("Ask")
        answer_md = gr.Markdown()
        sources_json = gr.JSON(label="Sources (if document-grounded)")

        def on_ask(q):
            q = (q or "").strip()
            if not q:
                return "Please enter a question.", []
            try:
                ans, srcs, used_rag = retrieve_and_answer(q, top_k=5)
                if used_rag and srcs:
                    ans += "\n\n**Sources:** " + ", ".join(f"[{s}]" for s in srcs)
                ans += "\n\n> " + DISCLAIMER
                return ans, (srcs if used_rag else [])
            except Exception as e:
                return (f"Sorry, something went wrong. Please try again or upload documents.\n\n"
                        f"_Error: {e}_\n\n> {DISCLAIMER}"), []

        ask_btn.click(fn=on_ask, inputs=question, outputs=[answer_md, sources_json])

    gr.Markdown("### Notes\n- Upload trusted, high-quality medical PDFs.\n- Without docs, answers are general info only.\n")

if __name__ == "__main__":
    try:
        init_embeddings(); init_pinecone()
    except Exception:
        pass
    demo.launch(server_name="0.0.0.0", server_port=int(os.getenv("PORT", "7860")))
