from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from langchain_unstructured import UnstructuredLoader
from crewai_tools import tool
from dotenv import load_dotenv
import os

load_dotenv()


@tool("Organization Knowledge")
def organization_knowledge(query: str) -> str:
    """
    Use this tool to fine information about the organization.
    """
    # Initialize Pinecone
    pinecone = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index = pinecone.Index("organization-knowledge")
    
