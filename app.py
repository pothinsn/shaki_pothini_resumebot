# Your Streamlit résumé chatbot app

import os, re, json
from io import BytesIO
from typing import List, Dict, Tuple
import numpy as np
import streamlit as st
from openai import OpenAI

try:
    import pypdf
except ImportError:
    pypdf = None

MODEL_NAME = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
EMBED_MODEL = os.environ.get("OPENAI_EMBED_MODEL", "text-embedding-3-small")

def clean_text(t: str) -> str:
    return re.sub(r"\s+", " ", (t or "")).strip()

def split_text(text: str, chunk_size=900, overlap=200) -> List[str]:
    text = clean_text(text)
    out, start = [], 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        out.append(text[start:end])
        if end == len(text): break
        start = max(0, end - overlap)
    return out

def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-10
    return float(np.dot(a, b) / denom)

def embed(client: OpenAI, texts: List[str]) -> List[List[float]]:
    if not texts: return []
    r = client.embeddings.create(model=EMBED_MODEL, input=texts)
    return [d.embedding for d in r.data]

def extract_pdf_text(file_bytes: bytes) -> str:
    if pypdf is None:
        raise RuntimeError("Install pypdf (see requirements.txt)")
    reader = pypdf.PdfReader(BytesIO(file_bytes))
    pages = []
    for pg in reader.pages:
        try:
            pages.append(pg.extract_text() or "")
        except Exception:
            pages.append("")
    return "\n".join(pages)

def build_index(client: OpenAI, text: str) -> Dict:
    chunks = split_text(text)
    vecs = embed(client, chunks)
    return {"chunks": chunks, "vecs": vecs}

def retrieve(index: Dict, query: str, client: OpenAI, k: int = 4) -> List[str]:
    if not index: return []
    qvec = embed(client, [query])[0]
    scores: List[Tuple[float, str]] = []
    for v, c in zip(index["vecs"], index["chunks"]):
        scores.append((cosine(np.array(qvec), np.array(v)), c))
    scores.sort(key=lambda x: x[0], reverse=True)
    return [c for _, c in scores[:k]]

def answer(client: OpenAI, query: str, contexts: List[str], tone: str, audience: str) -> str:
    system = (
        "You are a concise, recruiter-facing résumé assistant. "
        "Answer ONLY from the provided context; if not present, say so briefly. "
        "Prefer 3–5 bullets; keep under 120 words unless asked."
    )
    ctx = "\n\n".join([f"[Context {i+1}] {c}" for i, c in enumerate(contexts)])
    user = f"AUDIENCE: {audience}\nTONE: {tone}\nRésumé Context:\n{ctx}\n\nQuestion: {query}"
    r = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role":"system","content":system},{"role":"user","content":user}],
        temperature=0.2,
    )
    return r.choices[0].message.content.strip()

st.set_page_config(page_title="Chat with my Résumé", page_icon="🗂️", layout="centered")
st.title("🗂️ Chat with my Résumé")

with st.sidebar:
    st.subheader("Setup")
    api_key = st.text_input("OpenAI API Key", type="password")
    tone = st.selectbox("Tone", ["professional", "executive", "friendly", "enthusiastic"], index=0)
    audience = st.selectbox("Audience", ["General", "SWE", "PM", "Data", "AI/ML"], index=0)
    k = st.slider("Evidence passages", 1, 8, 4)

if "index" not in st.session_state:
    st.session_state.index = {}
if "messages" not in st.session_state:
    st.session_state.messages = []

uploaded = st.file_uploader("Upload your résumé (PDF).", type=["pdf"])

col1, col2 = st.columns(2)
with col1:
    build = st.button("📚 Build / Refresh Index", disabled=(not api_key or uploaded is None))
with col2:
    st.caption("Tip: Rebuild after uploading a new PDF.")

client = OpenAI(api_key=api_key) if api_key else None

if build:
    if client is None:
        st.error("Enter your OpenAI API key in the sidebar.")
    else:
        try:
            text = extract_pdf_text(uploaded.read())
            st.session_state.index = build_index(client, text)
            st.success(f"Index ready. {len(st.session_state.index.get('chunks', []))} passages.")
        except Exception as e:
            st.exception(e)

st.info("Try: “120-word leadership summary”, “3 bullets for Diagnostics PM role”, “Top outcomes from instrument control projects.”")

for m in st.session_state.messages:
    st.chat_message(m["role"]).markdown(m["content"])

q = st.chat_input("Ask about my résumé...")
if q:
    st.session_state.messages.append({"role":"user","content":q})
    st.chat_message("user").markdown(q)
    if client is None:
        st.chat_message("assistant").error("Add your API key in the sidebar.")
    elif not st.session_state.index:
        st.chat_message("assistant").error("Build the index first.")
    else:
        ctx = retrieve(st.session_state.index, q, client, k=k)
        a = answer(client, q, ctx, tone=tone, audience=audience)
        st.session_state.messages.append({"role":"assistant","content":a})
        st.chat_message("assistant").markdown(a)
        with st.expander("Sources used"):
            if ctx:
                for i, c in enumerate(ctx, 1):
                    st.write(f"**Source {i}:** {c[:600]}{'…' if len(c) > 600 else ''}")
            else:
                st.write("No sources.")
