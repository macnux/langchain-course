import json
import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

load_dotenv()

def ingest_data():
    # 1. Load the crawled data
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(BASE_DIR, "crawl_results.json")
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        print("Error: crawl_results.json not found. Please run crawl-docs.py first.")
        return

    results = data.get("results", [])
    print(f"Loaded {len(results)} pages from crawl_results.json")

    # 2. Convert to LangChain Documents
    documents = []
    for item in results:
        url = item.get("url")
        content = item.get("raw_content") or item.get("content") or ""
        
        if content:
            doc = Document(page_content=content, metadata={"source": url})
            documents.append(doc)

    print(f"Created {len(documents)} document objects")

    # 3. Split the text
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    
    chunks = text_splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks")

    # 4. Initialize Embeddings
    # Using configuration from rag.py for local LLM compatibility
    embeddings = OpenAIEmbeddings(
        model="text-embedding-embeddinggemma-300m",  
        openai_api_key="not-needed",
        base_url="http://localhost:1234/v1",
        check_embedding_ctx_length=False
    )

    # 5. Create and Save FAISS Index
    print("Creating FAISS index... (this may take a moment)")
    try:
        vectorstore = FAISS.from_documents(documents=chunks, embedding=embeddings)
        
        index_name = os.path.join(BASE_DIR, "faiss_index_pocketbase")
        vectorstore.save_local(index_name)
        print(f"Successfully saved FAISS index to folder: {index_name}")
        
    except Exception as e:
        print(f"Error creating/saving FAISS index: {e}")

if __name__ == "__main__":
    ingest_data()
