import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

def main():
    # 1. Setup Embeddings (Same as ingestion)
    embeddings = OpenAIEmbeddings(
        model="text-embedding-embeddinggemma-300m",
        openai_api_key="not-needed",
        base_url="http://localhost:1234/v1",
        check_embedding_ctx_length=False
    )

    # 2. Load FAISS Index
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    INDEX_NAME = os.path.join(BASE_DIR, "faiss_index_pocketbase")
    if not os.path.exists(INDEX_NAME):
        print(f"Error: Index {INDEX_NAME} not found. Run ingest_faiss.py first.")
        return

    try:
        vectorstore = FAISS.load_local(
            INDEX_NAME, 
            embeddings, 
            allow_dangerous_deserialization=True
        )
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    except Exception as e:
        print(f"Error loading index: {e}")
        return

    # 3. Setup Local LLM
    # User specified "openai/gpt-oss-20b"
    llm = ChatOpenAI(
        model="openai/gpt-oss-20b",
        openai_api_key="not-needed",
        base_url="http://localhost:1234/v1",
        temperature=0.7
    )

    # 4. Define Prompt Template
    template = """Answer the question based only on the following context:
{context}

Question: {question}
"""
    prompt = ChatPromptTemplate.from_template(template)

    # 5. Build Chain
    def format_docs(docs):
        return "\n\n".join([d.page_content for d in docs])

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    # 6. Interactive Loop
    print(f"RAG System Ready (Model: openai/gpt-oss-20b)")
    print("Type 'exit' to quit.")
    
    while True:
        query = input("\nEnter your question: ")
        if query.lower() in ["exit", "quit", "q"]:
            break
        
        if not query.strip():
            continue

        try:
            # We can also verify context retrieval
            # docs = retriever.invoke(query)
            # print(f"DEBUG: Retrieved {len(docs)} chunks")
            
            response = chain.invoke(query)
            print("-" * 50)
            print("Answer:")
            print(response)
            print("-" * 50)
        except Exception as e:
            print(f"Error generating answer: {e}")

if __name__ == "__main__":
    main()
