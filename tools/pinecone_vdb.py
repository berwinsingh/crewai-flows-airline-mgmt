from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from langchain_unstructured import UnstructuredLoader
from langchain_openai import OpenAIEmbeddings
from crewai_tools import tool
from dotenv import load_dotenv
import os


load_dotenv()
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
embeddings = OpenAIEmbeddings(model="text-embedding-3-large", api_key=os.getenv("OPENAI_API_KEY"))

index_name = "airline-knowledge"

@tool("Organization Knowledge")
def organization_knowledge(query: str, namespace: str = None) -> str:
    """
    Use this tool to find information about the organization.
    Optionally specify a namespace to search in a specific category like cancellation policies, baggage policies, refund policies, etc.
    """
    index = pc.Index(index_name)
    
    vector_store = PineconeVectorStore(index=index, embedding=embeddings)
    
    # Perform similarity search with namespace
    results = vector_store.similarity_search(
        query,
        k=3,
        namespace=namespace
    )
    
    formatted_results = []
    for i, doc in enumerate(results, 1):
        formatted_results.append(f"{i}. {doc.page_content}\n")
    
    response = "\n".join(formatted_results)
    
    namespace_info = f" in the '{namespace}' category" if namespace else ""
    return f"Here's what I found about your query{namespace_info}:\n\n{response}"

def create_index(directory_path=None, namespace=None):
    existing_indexes = pc.list_indexes()
    
    if index_name not in existing_indexes:
        print(f"Creating new index: {index_name}")
        pc.create_index(
            name=index_name,
            dimension=1536,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-west-2")
        )
        print(f"Index {index_name} created successfully")
    else:
        print(f"Index {index_name} already exists")
    
    index = pc.Index(index_name)
    
    if directory_path:
        print("Loading and processing documents")
        loader = UnstructuredLoader(directory_path)
        documents = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        split_docs = text_splitter.split_documents(documents)
        
        print(f"Adding documents to the index in namespace: {namespace}")
        vector_store = PineconeVectorStore(index=index, embedding=embeddings)
        vector_store.add_documents(split_docs, namespace=namespace)
        print("Documents added successfully")
    
    return index
