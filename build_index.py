# build_index.py
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import pandas as pd
import os

# make sure folders exist
os.makedirs("data", exist_ok=True)
os.makedirs("index", exist_ok=True)

# 1) read the finance sentences
df = pd.read_csv("data/finance_phrases.csv")
texts = df["text"].tolist()

# 2) create the embedding maker (small & fast)
embedder = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# 3) optional metadata (helps show sources)
metadatas = [{"row": i} for i in range(len(texts))]

# 4) build & persist the vector index (memory cupboard)
Chroma.from_texts(
    texts=texts,
    embedding=embedder,
    metadatas=metadatas,
    persist_directory="index"
)

print("✅ Index built and saved in /index")
