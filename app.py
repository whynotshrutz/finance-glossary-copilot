import os
from pathlib import Path
import streamlit as st
from openai import OpenAI
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

BASE_URL = os.getenv("OPENAI_BASE_URL", "http://localhost:11434/v1")
API_KEY = os.getenv("OPENAI_API_KEY", "ollama")
MODEL_NAME = os.getenv("OLLAMA_MODEL", "llama3.2")
INDEX_DIR = os.getenv("INDEX_DIR", "index")
EMB_MODEL = os.getenv("EMB_MODEL", "all-MiniLM-L6-v2")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

st.set_page_config(page_title="Finance Glossary Copilot", page_icon="💬", layout="wide")
st.markdown("""
<style>
.answer-box { background: #ffffff; padding: 18px; border-radius: 12px; box-shadow: 0 0 8px rgba(0,0,0,0.06); }
.src { font-size: 0.92rem; color: #333; }
.dim { color: #666; font-size: 0.9rem; }
</style>
""", unsafe_allow_html=True)
st.markdown("<h2 style='margin:0'>💬 Finance Glossary Copilot</h2>", unsafe_allow_html=True)
st.caption("Grounded Q&A over your CSV notes using a local LLM (Ollama).")

with st.sidebar:
    st.subheader("⚙️ Settings")
    st.write("Model in use:")
    st.code(f"{MODEL_NAME}", language="text")
    explain_simple = st.checkbox("Explain like I’m 15 ", value=False)

@st.cache_resource(show_spinner=False)
def load_db():
    embedder = HuggingFaceEmbeddings(model_name=EMB_MODEL)
    db = Chroma(persist_directory=INDEX_DIR, embedding_function=embedder)
    return db

def mmr_search(db, query, k=5, lambda_mult=0.4):
    try:
        docs = db.max_marginal_relevance_search(query, k=k, fetch_k=max(k * 3, 20), lambda_mult=lambda_mult)
    except Exception:
        docs = db.similarity_search(query, k=k)
    seen, deduped = set(), []
    for d in docs:
        key = d.page_content.strip()[:120].lower()
        if key not in seen:
            seen.add(key)
            deduped.append(d)
    return deduped[:k]

def build_prompt(context_text, question, eli15):
    style = "Explain in simple terms, suitable for a 15-year-old. Use short sentences and a small example. " if eli15 else "Be concise and correct. "
    return ("You are a careful finance tutor. Use ONLY the CONTEXT to answer. If the context is insufficient, reply: 'Not found in my notes yet.' "
            + style + "\n\nCONTEXT:\n" + context_text + "\n\nQUESTION: " + question + "\n\nANSWER:")

query = st.text_input("🔎 Ask a finance question (e.g., 'What is diversification?')")

if query:
    db = load_db()
    with st.spinner("Searching notes…"):
        docs = mmr_search(db, query)
    if not docs:
        st.info("Not found in my notes yet. Try adding more definitions.", icon="ℹ️")
        st.stop()
    context = "\n".join(d.page_content for d in docs)
    prompt = build_prompt(context, query, explain_simple)
    client = OpenAI(base_url=BASE_URL, api_key=API_KEY)
    with st.spinner("Thinking with local model…"):
        try:
            resp = client.chat.completions.create(model=MODEL_NAME, messages=[{"role": "user", "content": prompt}])
            answer = resp.choices[0].message.content
        except Exception as e:
            st.error("Could not reach the local model. Is Ollama running?")
            st.code(f"ollama pull {MODEL_NAME}")
            st.exception(e)
            st.stop()
    st.markdown("<div class='answer-box'>", unsafe_allow_html=True)
    st.subheader("💡 Answer")
    st.write(answer.strip() if answer else "Not found in my notes yet.")
    st.markdown("</div>", unsafe_allow_html=True)
    st.divider()
    st.subheader("📘 Sources (top matches)")
    for i, d in enumerate(docs, 1):
        meta = d.metadata or {}
        row = meta.get("row", "—")
        src = meta.get("source", None)
        preview = d.page_content.strip().replace("\n", " ")
        src_label = f" • source: **{src}**" if src else ""
        st.markdown(f"<div class='src'>**{i}. Row {row}**{src_label}<br>{preview[:220]}{'…' if len(preview) > 220 else ''}</div>", unsafe_allow_html=True)
else:
    st.markdown("<p class='dim'>Tip: add clear definitions to your CSV and rebuild the index for sharper answers.</p>", unsafe_allow_html=True)
