# 💰 Finance Glossary Copilot

**Live App:** [https://finance-glossary-copilot.streamlit.app](https://finance-glossary-copilot.streamlit.app)

---

## 💡 What is this project?

**Finance Glossary Copilot** is a simple web app that helps users understand **financial terms and concepts** in an easy and interactive way.  
You can type any finance word (like *“Inflation”*, *“EBITDA”*, or *“Derivatives”*) and the app will explain it clearly — powered by AI.

It’s built to make **learning finance simpler for beginners** and act as a quick **reference tool for professionals**.

---

## ⚙️ Technologies Used

- **Python** — core language  
- **Streamlit** — for the web app interface  
- **OpenAI API** — for generating explanations  
- **LangChain** — for connecting LLMs with data  
- **ChromaDB** — for storing and retrieving finance term embeddings  
- **Hugging Face Embeddings** — to understand term meanings better  
- **Docker** — for easy deployment and containerization  

---

## 🚀 How it Works

1. The app reads a dataset of financial terms and builds a small knowledge index.  
2. When you search or ask a question, it retrieves the most relevant info using embeddings.  
3. The OpenAI model then generates a clear and simple explanation.  

---

## 💻 Run Locally

If you want to try it on your own system:
```bash
git clone https://github.com/whynotshrutz/finance-glossary-copilot.git
cd finance-glossary-copilot
pip install -r requirements.txt
streamlit run app.py
