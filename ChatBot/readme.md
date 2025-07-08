# 🤖 Semantic Chatbot using BERT, FAISS & Flan-T5

This project is a simple, fast, and effective chatbot that answers user questions by retrieving the most relevant information from a custom text file (`Combined_data.txt`) using semantic search (BERT + FAISS), and generates natural language responses using a generative model (Flan-T5).

---

## 🔍 What It Does

1. **Reads Knowledge**: Loads text data from `Combined_data.txt`.
2. **Embeds Semantics**: Uses BERT embeddings (via SentenceTransformers) to vectorize the text.
3. **Indexes for Speed**: Builds a FAISS index to enable fast similarity-based search.
4. **Retrieves Context**: On each question, it finds the top relevant chunks from the document.
5. **Generates Answers**: Feeds the context + question to Google Flan-T5 to generate a coherent answer.
6. **Interactive UI**: Runs on Streamlit, allowing real-time Q&A in your browser.

---

## 🛠️ Technologies Used

- [Streamlit](https://streamlit.io/) – for the UI
- [SentenceTransformers](https://www.sbert.net/) – for BERT-based sentence embeddings
- [FAISS](https://github.com/facebookresearch/faiss) – for fast similarity search
- [Transformers](https://huggingface.co/transformers/) – to load Google Flan-T5 for answer generation
- [PyTorch](https://pytorch.org/) – backend for models

---

## How to use it (Locally)?

- Project Structure 
	|- Folder 
		|-bot.py
		|-Combined_data.txt
		|-readme.md
		|-requirements.txt

- Download the Required dependencies using the command - pip install -r requirements.txt
- Open the terminal in "Folder" and run the bot using the command - streamlit run bot.py
		
