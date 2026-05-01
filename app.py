import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import FakeEmbeddings
from langchain.vectorstores import Chroma

# ----------------------------
# PAGE CONFIG
# ----------------------------
st.set_page_config(page_title="Scholarship Finder", layout="wide")

st.title("🎓 Scholarship & Financial Aid Finder")

# ----------------------------
# SELECT SOURCE
# ----------------------------
source = st.radio("Select Source:", ["CSV Dataset", "Website"])

documents = []

# ----------------------------
# CSV MODE
# ----------------------------
if source == "CSV Dataset":
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)

        for _, row in df.iterrows():
            text = " ".join([str(v) for v in row.values])
            documents.append(Document(page_content=text))

        st.success("CSV Loaded!")

# ----------------------------
# WEBSITE MODE
# ----------------------------
if source == "Website":
    url = st.text_input("Enter Website URL")

    if url:
        try:
            response = requests.get(url)
            soup = BeautifulSoup(response.text, "html.parser")

            text = soup.get_text()

            documents.append(Document(page_content=text))

            st.success("Website Loaded!")

        except:
            st.error("Failed to load website")

# ----------------------------
# CREATE VECTOR DB
# ----------------------------
def create_db(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

    split_docs = splitter.split_documents(docs)

    embeddings = FakeEmbeddings(size=384)  # lightweight, no torch

    db = Chroma.from_documents(split_docs, embeddings)

    return db

# ----------------------------
# QUERY
# ----------------------------
if documents:
    db = create_db(documents)

    query = st.text_input("Ask a question")

    if query:
        results = db.similarity_search(query, k=3)

        st.subheader("Results:")
        for res in results:
            st.write(res.page_content[:500])

