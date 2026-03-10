from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Load HuggingFace embedding model (downloads once, cached after)
def get_embeddings():
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )
    return embeddings


def create_vectorstore(chunks):
    """
    Take document chunks → embed them → store in FAISS
    Returns a FAISS vectorstore object
    """
    embeddings = get_embeddings()
    
    vectorstore = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings
    )
    
    print(f"✅ Vector store created with {len(chunks)} chunks")
    return vectorstore


def get_retriever(vectorstore, k=4):
    """
    Returns a retriever that fetches top-k relevant chunks
    for a given query
    k=4 means → return 4 most relevant chunks
    """
    retriever = vectorstore.as_retriever(
        search_kwargs={"k": k}
    )
    return retriever
