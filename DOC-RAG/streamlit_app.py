import streamlit as st
import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

st.set_page_config(page_title="PocketBase RAG", page_icon="📝")

# Custom CSS for Facebook-like look
st.markdown("""
<style>
    /* Font family */
    body {
        font-family: Helvetica, Arial, sans-serif;
    }
    
    /* Input fields and buttons rounded corners */
    .stTextInput > div > div > input, .stTextArea > div > div > textarea {
        border-radius: 8px;
        background-color: #FFFFFF;
        border: 1px solid #dddfe2;
    }
    
    /* Submit button styling */
    .stButton > button {
        background-color: #1877F2;
        color: white;
        border-radius: 6px;
        font-weight: bold;
        border: none;
        padding: 0.5rem 1rem;
    }
    .stButton > button:hover {
        background-color: #166fe5;
        color: white;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #FFFFFF;
        border-right: 1px solid #dddfe2;
    }
    
    /* Titles and Headers */
    h1, h2, h3 {
        color: #1877F2;
    }
</style>
""", unsafe_allow_html=True)


st.title("PocketBase Documentation RAG")
st.write("Ask questions about PocketBase documentation using your local LLM.")

# Sidebar - User Profile
with st.sidebar:
    st.image("https://ui-avatars.com/api/?name=User+Name&background=random", width=100)
    st.title("User Profile")
    st.write("**Name:** User Name")
    st.write("**Email:** user@example.com")
    st.write("**Role:** Developer")
    st.divider()
    st.write("Logged in via: Local Session")


# Setup paths - using absolute path relative to this script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INDEX_PATH = os.path.join(BASE_DIR, "faiss_index_pocketbase")

# Initialize LLM and Embeddings (cached to avoid reload)
@st.cache_resource
def get_chain():
    # Embeddings
    embeddings = OpenAIEmbeddings(
        model="text-embedding-embeddinggemma-300m",
        openai_api_key="not-needed",
        base_url="http://localhost:1234/v1",
        check_embedding_ctx_length=False
    )
    
    # Load Index
    if not os.path.exists(INDEX_PATH):
        return None
        
    try:
        vectorstore = FAISS.load_local(
            INDEX_PATH, 
            embeddings, 
            allow_dangerous_deserialization=True
        )
    except Exception as e:
        st.error(f"Error loading FAISS index: {e}")
        return None
        
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # LLM
    llm = ChatOpenAI(
        model="openai/gpt-oss-20b",
        openai_api_key="not-needed",
        base_url="http://localhost:1234/v1",
        temperature=0.7
    )

    # Chain
    template = """Answer the question based only on the following context:
{context}

Question: {question}
"""
    prompt = ChatPromptTemplate.from_template(template)

    def format_docs(docs):
        return "\n\n".join([d.page_content for d in docs])

    rag_chain_from_docs = (
        RunnablePassthrough.assign(context=(lambda x: format_docs(x["context"])))
        | prompt
        | llm
        | StrOutputParser()
    )

    rag_chain_with_source = RunnableParallel(
        {"context": retriever, "question": RunnablePassthrough()}
    ).assign(answer=rag_chain_from_docs)

    return rag_chain_with_source


chain = get_chain()

if chain is None:
    st.error(f"Index not found or could not be loaded at {INDEX_PATH}. Please make sure you have run 'ingest_faiss.py'.")
else:
    # Input
    query = st.text_area("Enter your question:", height=100)
    
    if st.button("Submit"):
        if query:
            with st.spinner("Thinking..."):
                try:
                    result = chain.invoke(query)
                    st.markdown("### Answer")
                    st.markdown(result["answer"])
                    
                    # Display sources
                    st.markdown("### Sources")
                    sources = set([doc.metadata.get("source") for doc in result["context"]])
                    for source in sources:
                        st.markdown(f"- [{source}]({source})")
                except Exception as e:
                    st.error(f"Error generating answer: {e}")
        else:
            st.warning("Please enter a question!")
