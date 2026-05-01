import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# -------------------------
# PAGE CONFIG
# -------------------------
st.set_page_config(page_title="🎓 Scholarship Finder", layout="wide")
st.title("🎓 Scholarship & Financial Aid Finder")

# -------------------------
# LOAD CSV DATA
# -------------------------
def load_csv():
    try:
        df = pd.read_csv("scholarship_data.csv")
        docs = []

        for _, row in df.iterrows():
            content = " ".join([str(i) for i in row.values])
            docs.append(Document(page_content=content))

        return docs
    except:
        st.error("❌ CSV not found!")
        return []

# -------------------------
# LOAD WEBSITE DATA
# -------------------------
def load_website(url):
    try:
        res = requests.get(url)
        soup = BeautifulSoup(res.text, "html.parser")

        # Clean text
        for script in soup(["script", "style"]):
            script.extract()

        text = soup.get_text(separator=" ")
        return [Document(page_content=text)]
    except:
        st.error("❌ Failed to load website")
        return []

# -------------------------
# CREATE VECTOR DB
# -------------------------
@st.cache_resource
def create_db(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=100
    )

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    db = Chroma.from_documents(
        splitter.split_documents(docs),
        embedding=embeddings
    )

    return db

# -------------------------
# UI
# -------------------------
source = st.radio("Select Source:", ["CSV Dataset", "Website"])

documents = []

if source == "CSV Dataset":
    documents = load_csv()

elif source == "Website":
    url = st.text_input("Enter Website URL")
    if url:
        documents = load_website(url)

# -------------------------
# SEARCH
# -------------------------
if documents:
    db = create_db(documents)

    query = st.text_input("Ask your question")

    if query:
        # Better retrieval (more + diverse)
        results = db.max_marginal_relevance_search(query, k=5)

        st.subheader("🔍 Top Matches:")
        for i, res in enumerate(results):
            st.write(f"**Result {i+1}:**")
            st.write(res.page_content[:500] + "...")
            st.divider()

        # -------------------------
        # SIMPLE ANSWER GENERATION
        # -------------------------
        combined_text = " ".join([doc.page_content for doc in results])

        st.subheader("💡 Final Answer:")
        st.write(combined_text[:1000] + "...")
