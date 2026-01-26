from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import os

load_dotenv()

# Initialize embeddings (must match the model used during ingestion)
embeddings = OpenAIEmbeddings(
    model="text-embedding-embeddinggemma-300m",
    openai_api_key="not-needed",
    base_url="http://localhost:1234/v1",
    check_embedding_ctx_length=False
)

# Initialize LLM (using LM Studio)
llm = ChatOpenAI(
    model="openai/gpt-oss-20b",  # Update this to your LM Studio model name
    openai_api_key="not-needed",
    base_url="http://localhost:1234/v1",
    temperature=0.7
)

# Global vectorstore variable
vectorstore = None
retriever = None


def load_pdf(pdf_path: str, chunk_size: int = 1000, chunk_overlap: int = 200):
    """
    Load a PDF file and create a FAISS vector store.
    
    Args:
        pdf_path: Path to the PDF file
        chunk_size: Size of text chunks (default: 1000)
        chunk_overlap: Overlap between chunks (default: 200)
    
    Returns:
        FAISS vectorstore
    """
    global vectorstore, retriever
    
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    
    print(f"📄 Loading PDF: {pdf_path}")
    
    # Load PDF
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    
    print(f"📑 Loaded {len(documents)} pages")
    
    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    splits = text_splitter.split_documents(documents)
    
    print(f"✂️ Created {len(splits)} text chunks")
    
    # Create FAISS vector store (uses GPU if faiss-gpu is installed)
    print("🔄 Creating FAISS vector store with embeddings...")
    vectorstore = FAISS.from_documents(splits, embeddings)
    
    # Create retriever
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
    print("PDF loaded and indexed successfully!")
    
    return vectorstore


def save_vectorstore(save_path: str = "faiss_index"):
    """
    Save the FAISS vector store to disk.
    
    Args:
        save_path: Directory path to save the index
    """
    global vectorstore
    if vectorstore is None:
        raise ValueError("No vectorstore loaded. Load a PDF first.")
    
    vectorstore.save_local(save_path)
    print(f"💾 Vector store saved to: {save_path}")


def load_vectorstore(load_path: str = "faiss_index"):
    """
    Load a FAISS vector store from disk.
    
    Args:
        load_path: Directory path to load the index from
    """
    global vectorstore, retriever
    
    if not os.path.exists(load_path):
        raise FileNotFoundError(f"FAISS index not found: {load_path}")
    
    vectorstore = FAISS.load_local(
        load_path, 
        embeddings,
        allow_dangerous_deserialization=True
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    print(f"📂 Vector store loaded from: {load_path}")


# Customizable prompt template for PDF chat
PROMPT_TEMPLATE = """You are a helpful assistant that answers questions based on the provided PDF content.
Use the following context from the PDF to answer the question.
If you don't know the answer based on the context, say so honestly.

Context from PDF:
{context}

Question: {question}

Answer:"""

prompt = PromptTemplate(
    template=PROMPT_TEMPLATE,
    input_variables=["context", "question"]
)


def format_docs(docs):
    """Format retrieved documents into a single string."""
    formatted = []
    for i, doc in enumerate(docs, 1):
        page_num = doc.metadata.get('page', 'Unknown')
        formatted.append(f"[Page {page_num}]\n{doc.page_content}")
    return "\n\n---\n\n".join(formatted)


def get_rag_chain():
    """Get the RAG chain for question answering."""
    global retriever
    if retriever is None:
        raise ValueError("No PDF loaded. Please load a PDF first using load_pdf()")
    
    return (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )


def search_similar(query: str, k: int = 3):
    """
    Perform similarity search on the vector store.
    
    Args:
        query: The search query text
        k: Number of similar documents to return (default: 3)
    
    Returns:
        List of similar documents
    """
    global vectorstore
    if vectorstore is None:
        raise ValueError("No PDF loaded. Please load a PDF first.")
    
    results = vectorstore.similarity_search(query, k=k)
    return results


def search_with_scores(query: str, k: int = 3):
    """
    Perform similarity search and return documents with their similarity scores.
    
    Args:
        query: The search query text
        k: Number of similar documents to return (default: 3)
    
    Returns:
        List of tuples (document, score)
    """
    global vectorstore
    if vectorstore is None:
        raise ValueError("No PDF loaded. Please load a PDF first.")
    
    results = vectorstore.similarity_search_with_score(query, k=k)
    return results


def ask(question: str) -> str:
    """
    Ask a question and get an AI-generated response based on retrieved context.
    
    Args:
        question: The question to ask
    
    Returns:
        AI-generated response
    """
    rag_chain = get_rag_chain()
    return rag_chain.invoke(question)


def chat_with_pdf():
    """
    Interactive chat loop with the loaded PDF.
    """
    print("\n" + "=" * 60)
    print("💬 PDF CHAT MODE")
    print("=" * 60)
    print("Type your questions about the PDF. Type 'quit' or 'exit' to stop.")
    print("Type 'sources' to see source chunks for your last query.")
    print("=" * 60 + "\n")
    
    last_query = None
    
    while True:
        try:
            query = input("You: ").strip()
            
            if not query:
                continue
            
            if query.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Goodbye!")
                break
            
            if query.lower() == 'sources' and last_query:
                print("\n📚 Source chunks for last query:")
                print("-" * 40)
                results = search_similar(last_query, k=3)
                for i, doc in enumerate(results, 1):
                    page = doc.metadata.get('page', 'Unknown')
                    print(f"\n[Chunk {i} - Page {page}]")
                    print(doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content)
                print("-" * 40 + "\n")
                continue
            
            last_query = query
            
            print("\n🤖 Assistant: ", end="", flush=True)
            response = ask(query)
            print(response)
            print()
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}\n")


if __name__ == "__main__":
    print("=" * 60)
    print("📚 PDF CHAT WITH FAISS-GPU")
    print("=" * 60)
    
    # Check for existing FAISS index
    if os.path.exists("faiss_index"):
        choice = input("\n📂 Found existing FAISS index. Load it? (y/n): ").strip().lower()
        if choice == 'y':
            load_vectorstore("faiss_index")
        else:
            pdf_path = "react.pdf"
            if os.path.exists(pdf_path):
                 load_pdf(pdf_path)
            else:
                 print(f"❌ Default file '{pdf_path}' not found.")
                 pdf_path = input("\n📄 Enter the path to your PDF file: ").strip()
                 load_pdf(pdf_path)
            
            save_choice = input("\n💾 Save the index for future use? (y/n): ").strip().lower()
            if save_choice == 'y':
                save_vectorstore("faiss_index")
    else:
        pdf_path = "react.pdf"
        if os.path.exists(pdf_path):
             load_pdf(pdf_path)
        else:
             print(f"❌ Default file '{pdf_path}' not found.")
             pdf_path = input("\n📄 Enter the path to your PDF file: ").strip()
             load_pdf(pdf_path)
        
        save_choice = input("\n💾 Save the index for future use? (y/n): ").strip().lower()
        if save_choice == 'y':
            save_vectorstore("faiss_index")
    
    # Start interactive chat
    chat_with_pdf()
