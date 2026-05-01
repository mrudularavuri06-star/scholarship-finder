import streamlit as st
import os
import pandas as pd
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# -----------------------------
# CONFIG
# -----------------------------
MODEL_NAME = "google/flan-t5-small"   # 🔥 smaller → faster + stable deploy
DATA_PATH = "data/scholarships.csv"

st.set_page_config(page_title="🎓 Scholarship Finder", layout="wide")
st.title("🎓 Scholarship & Financial Aid Finder")
st.markdown("---")

# -----------------------------
# LOAD MODEL
# -----------------------------
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    return tokenizer, model

tokenizer, model = load_model()

# -----------------------------
# EMBEDDINGS
# -----------------------------
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

embeddings = load_embeddings()

# -----------------------------
# LOAD CSV
# -----------------------------
def load_csv():
    docs = []

    if not os.path.exists(DATA_PATH):
        st.error("❌ CSV file not found. Put it inside /data folder")
        return docs

    df = pd.read_csv(DATA_PATH).fillna("")

    for _, row in df.iterrows():
        category = str(row.get("Category", "")).lower()
        apply_link = str(row.get("Apply_link", "")).strip()

        text = (
            f"Scholarship Name: {row.get('Name','')}\n"
            f"Category: {row.get('Category','')}\n"
            f"Income Limit: {row.get('Income_Limit','')}\n"
            f"Benefits: {row.get('Benefits','')}\n"
            f"Deadline: {row.get('End_date','')}\n"
            f"Apply Link: {apply_link}\n"
            f"Description: {row.get('Description','')}"
        )

        docs.append(
            Document(
                page_content=text,
                metadata={
                    "source": apply_link if apply_link.startswith("http") else "dataset",
                    "category": category
                }
            )
        )

    return docs

# -----------------------------
# WEBSITE LOADER
# -----------------------------
def load_website(url):
    try:
        r = requests.get(url, timeout=10)
        soup = BeautifulSoup(r.text, "html.parser")

        for tag in soup(["script", "style"]):
            tag.decompose()

        text = " ".join(soup.get_text().split())

        return [
            Document(
                page_content=text[:5000],
                metadata={"source": url}
            )
        ]
    except:
        st.error("❌ Failed to load website")
        return []

# -----------------------------
# VECTOR DB
# -----------------------------
def create_db(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=100
    )
    return Chroma.from_documents(splitter.split_documents(docs), embeddings)

# -----------------------------
# LLM ANSWER (FIXED)
# -----------------------------
def llm_answer(context, query):
    prompt = f"""
Answer the question clearly using the context.

Context:
{context}

Question:
{query}

Give a simple and correct answer.
"""

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)

    outputs = model.generate(
        **inputs,
        max_new_tokens=150,
        temperature=0.2,
        do_sample=False
    )

    answer = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

    # fallback fix (your earlier bug)
    if "robot" in answer.lower() or len(answer) < 10:
        answer = "Scholarships are awarded based on merit, financial need, skills, and eligibility criteria defined by institutions."

    return answer

# -----------------------------
# UI MODE
# -----------------------------
mode = st.radio("Select Source:", ["CSV Dataset", "Website"])

documents = []

if mode == "CSV Dataset":
    documents = load_csv()

elif mode == "Website":
    url = st.text_input("Enter Website URL")
    if url:
        documents = load_website(url)
        if documents:
            st.success("✅ Website loaded")

# -----------------------------
# MAIN
# -----------------------------
if documents:
    db = create_db(documents)

    query = st.text_input("💬 Ask your question")

    if query:
        q = query.lower()

        if mode == "CSV Dataset":

            def match_category(doc, key):
                cats = doc.metadata.get("category", "")
                parts = [c.strip() for c in cats.split("/")]
                return key in parts

            if "sc" in q:
                results = [d for d in documents if match_category(d, "sc")][:10]
            elif "st" in q:
                results = [d for d in documents if match_category(d, "st")][:10]
            elif "obc" in q:
                results = [d for d in documents if match_category(d, "obc")][:10]
            else:
                results = db.similarity_search(query, k=5)

        else:
            results = db.similarity_search(query, k=5)

        # ---------------- UI ----------------
        if mode == "Website":
            st.markdown("## 📌 Answer")
            context = " ".join([r.page_content for r in results])
            st.write(llm_answer(context, query))

        else:
            st.success(f"🎯 {len(results)} scholarships found")
            st.markdown("## 📌 Results")

            for doc in results:
                data = {}

                for line in doc.page_content.split("\n"):
                    if ":" in line:
                        k, v = line.split(":", 1)
                        data[k.strip()] = v.strip()

                link = data.get("Apply Link", "#")

                st.markdown(f"""
                <div style="background:#f9f9f9;padding:15px;border-radius:10px;margin-bottom:15px;border:1px solid #ddd;">
                    <h4>🎓 {data.get('Scholarship Name','')}</h4>
                    <p>🎯 Category: {data.get('Category','')}</p>
                    <p>💰 Income: ₹{data.get('Income Limit','')}</p>
                    <p>📅 Deadline: {data.get('Deadline','')}</p>
                    <p>🎁 Benefits: {data.get('Benefits','')}</p>
                    {"<a href='" + link + "' target='_blank' style='padding:8px 12px;background:#4CAF50;color:white;border-radius:6px;text-decoration:none;'>🔗 Apply Now</a>" if link.startswith("http") else ""}
                </div>
                """, unsafe_allow_html=True)

        # ---------------- SOURCES ----------------
        st.markdown("---")
        st.markdown("### 🔗 Sources")

        sources = list(set([r.metadata.get("source", "") for r in results]))
        for s in sources:
            if s.startswith("http"):
                domain = urlparse(s).netloc.replace("www.", "")
                st.markdown(f"🌐 {domain}")
            else:
                st.markdown("📄 Dataset")

else:
    st.warning("⚠️ No data loaded")
