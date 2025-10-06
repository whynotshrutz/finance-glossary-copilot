from datasets import load_dataset
import pandas as pd

# 1) download a small finance dataset
ds = load_dataset("financial_phrasebank", "sentences_50agree")


# 2) pick first 200 sentences (nice and tiny)
rows = ds["train"].select(range(200))
data = [item["sentence"] for item in rows]

# 3) save to CSV in /data
pd.DataFrame({"text": data}).to_csv("data/finance_phrases.csv", index=False)

print("✅ Saved 200 rows to data/finance_phrases.csv")
