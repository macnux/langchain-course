from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader

from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import CharacterTextSplitter

load_dotenv()

if __name__ == "__main__":
    print("Ingesting...")
    loader = TextLoader("mediumblog1.txt", encoding="utf-8")
    document = loader.load()

    print("splitting...")
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(document)
    print(f"created {len(texts)} chunks")

    embeddings = OpenAIEmbeddings(
        model="text-embedding-embeddinggemma-300m",  # Update this to match your LM Studio model name
        openai_api_key="not-needed",
        base_url="http://localhost:1234/v1",
        check_embedding_ctx_length=False  # Prevents input format issues with LM Studio
    )

    print("ingesting...")
    PineconeVectorStore.from_documents(
        texts, embeddings, index_name="medium-blogs-embeddings-index"
    )