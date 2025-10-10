import os
import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from openai import OpenAI

INDEX_DIR       = os.getenv("INDEX_DIR", "index")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "glossary_v1")
EMB_MODEL       = os.getenv("EMB_MODEL", "all-MiniLM-L6-v2")

USE_LLM         = os.getenv("USE_LLM", "false").lower() == "true"
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "http://localhost:11434/v1")
OPENAI_API_KEY  = os.getenv("OPENAI_API_KEY", "ollama")
MODEL_NAME      = os.getenv("MODEL_NAME", "llama3.2")

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

st.set_page_config(page_title="Finance Glossary Copilot", page_icon="💬", layout="wide")
st.markdown("""
<style>
.answer-box { background:#fff; padding:18px; border-radius:12px; box-shadow:0 0 8px rgba(0,0,0,0.06); }
.src { font-size:0.92rem; color:#333; }
.dim { color:#666; font-size:0.9rem; }
</style>
""", unsafe_allow_html=True)
st.markdown("<h2 style='margin:0'>💬 Finance Glossary Copilot</h2>", unsafe_allow_html=True)
st.caption("Grounded Q&A over your CSV notes. Retrieval-only by default; optional LLM generation when enabled.")

with st.sidebar:
    st.subheader("⚙️ Settings")
    st.code(f"USE_LLM={USE_LLM} · MODEL={MODEL_NAME}", language="text")
    st.code(f"{INDEX_DIR} / {COLLECTION_NAME}", language="text")
    explain_simple = st.checkbox("Explain like I’m 15", value=False)

@st.cache_resource(show_spinner=False)
def load_db():
    embedder = HuggingFaceEmbeddings(model_name=EMB_MODEL)
    return Chroma(persist_directory=INDEX_DIR, collection_name=COLLECTION_NAME, embedding_function=embedder)

def mmr_search(db, query, k=5, lambda_mult=0.4):
    try:
        docs = db.max_marginal_relevance_search(query, k=k, fetch_k=max(k * 3, 20), lambda_mult=lambda_mult)
    except Exception:
        docs = db.similarity_search(query, k=k)
    seen, out = set(), []
    for d in docs:
        key = (d.page_content or "").strip()[:160].lower()
        if key and key not in seen:
            seen.add(key)
            out.append(d)
    return out[:k]

def build_prompt(context_text: str, question: str, eli15: bool) -> str:
    style = "Explain simply for a 15-year-old. Short sentences and one small example. " if eli15 else "Be concise and precise. "
    return (
        "Answer ONLY using the CONTEXT. If not in context, reply: 'Not found in my notes yet.' "
        + style + "\n\nCONTEXT:\n" + context_text + "\n\nQUESTION: " + question + "\n\nANSWER:"
    )

@st.cache_resource(show_spinner=False)
def get_llm_client():
    try:
        return OpenAI(base_url=OPENAI_BASE_URL, api_key=OPENAI_API_KEY)
    except Exception:
        return None

def generate_with_llm(prompt: str) -> str | None:
    client = get_llm_client()
    if not client:
        return None
    try:
        r = client.chat.completions.create(model=MODEL_NAME, messages=[{"role": "user", "content": prompt}], temperature=0.0)
        return (r.choices[0].message.content or "").strip()
    except Exception:
        return None

query = st.text_input("🔎 Ask a finance question (e.g., 'What is diversification?')")

# quick index sanity
try:
    _ = load_db().similarity_search("peek", k=1)
    st.sidebar.caption("Index: loaded")
except Exception as e:
    st.sidebar.error("Index not found/unreadable. Check INDEX_DIR and COLLECTION_NAME.")
    st.sidebar.exception(e)

if query:
    db = load_db()
    with st.spinner("Searching…"):
        docs = mmr_search(db, query, k=5)
    if not docs:
        st.info("Not found in my notes yet. Add more definitions or verify collection name.", icon="ℹ️")
        st.stop()

    context_text = "\n---\n".join((d.page_content or "").strip() for d in docs if d and d.page_content)

    if not USE_LLM:
        answer = docs[0].page_content.strip()
    else:
        prompt = build_prompt(context_text, query, explain_simple)
        with st.spinner("Thinking…"):
            answer = generate_with_llm(prompt) or docs[0].page_content.strip()

    st.markdown("<div class='answer-box'>", unsafe_allow_html=True)
    st.subheader("💡 Answer")
    st.write(answer if answer else "Not found in my notes yet.")
    st.markdown("</div>", unsafe_allow_html=True)

    st.divider()
    st.subheader("📘 Sources (top matches)")
    for i, d in enumerate(docs, 1):
        meta = d.metadata or {}
        row = meta.get("row", "—")
        src = meta.get("source", None)
        preview = (d.page_content or "").strip().replace("\n", " ")
        src_label = f" • source: **{src}**" if src else ""
        st.markdown(
            f"<div class='src'>**{i}. Row {row}**{src_label}<br>{preview[:220]}{'…' if len(preview) > 220 else ''}</div>",
            unsafe_allow_html=True
        )
else:
    st.markdown("<p class='dim'>Tip: keep CSV definitions clear; rebuild the index after edits.</p>", unsafe_allow_html=True)
