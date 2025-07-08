import streamlit as st
import torch
import faiss
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# === CONFIGURATION ===
DATA_FILE = 'Combined_data.txt'
EMBED_MODEL = 'all-MiniLM-L6-v2'
GEN_MODEL = 'google/flan-t5-base'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# === Load models (no caching to avoid pickle error) ===
def load_models():
    embedder = SentenceTransformer(EMBED_MODEL)
    tokenizer = AutoTokenizer.from_pretrained(GEN_MODEL)
    generator = AutoModelForSeq2SeqLM.from_pretrained(GEN_MODEL).to(DEVICE)
    return embedder, tokenizer, generator

@st.cache_resource
def load_sentences(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        raw = f.read()
    return [line.strip() for line in raw.split('\n') if line.strip()]

def build_faiss_index(sentences, embedder):
    embeddings = embedder.encode(sentences, convert_to_numpy=True, show_progress_bar=True)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index, embeddings

# === Retrieval + Generation ===
def query_and_generate(question, index, sentences, embedder, tokenizer, generator):
    query_embedding = embedder.encode([question])
    D, I = index.search(query_embedding, k=3)
    top_chunks = [sentences[i] for i in I[0]]
    context = " ".join(top_chunks)

    prompt = f"Context: {context}\nQuestion: {question}\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(DEVICE)
    outputs = generator.generate(**inputs, max_length=128)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return answer, top_chunks

# === Streamlit UI ===
st.set_page_config(page_title="Semantic Chatbot", layout="wide")
st.title("🤖 Semantic Chatbot")
st.markdown("Ask questions based on `Combined_data.txt` knowledge base")

# Load everything
with st.spinner("Loading models and building index..."):
    embedder, tokenizer, generator = load_models()
    sentences = load_sentences(DATA_FILE)
    index, _ = build_faiss_index(sentences, embedder)

# User Input
user_input = st.text_input("Ask your question:", placeholder="e.g., What is the ICS?")
if user_input:
    with st.spinner("Generating answer..."):
        answer, sources = query_and_generate(user_input, index, sentences, embedder, tokenizer, generator)

    st.markdown("### 🧠 Answer")
    st.write(answer)

    with st.expander("🔎 Context used"):
        for i, chunk in enumerate(sources):
            st.markdown(f"**Chunk {i+1}:** {chunk}")

st.markdown("---")
st.caption("Built using BERT, FAISS & Flan-T5")
