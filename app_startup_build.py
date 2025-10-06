
import os, pandas as pd
from pathlib import Path
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

def ensure_index():
    Path("index").mkdir(exist_ok=True)
    # if index folder empty, build
    if not any(Path("index").iterdir()):
        dfs = []
        if Path("data/finance_phrases.csv").exists():
            dfs.append(pd.read_csv("data/finance_phrases.csv"))
        if Path("data/finance_glossary.csv").exists():
            dfs.append(pd.read_csv("data/finance_glossary.csv"))
        if not dfs:
            print("No data files found in /data. Skipping index build.")
            return
        df = pd.concat(dfs, ignore_index=True).dropna().drop_duplicates(subset=["text"])
        embedder = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        Chroma.from_texts(
            texts=df["text"].tolist(),
            embedding=embedder,
            persist_directory="index"
        )
        print("✅ Built index with", len(df), "rows")

if __name__ == "__main__":
    ensure_index()
