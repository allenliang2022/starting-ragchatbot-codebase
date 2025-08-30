import os
from dataclasses import dataclass
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

@dataclass
class Config:
    """Configuration settings for the RAG system"""
    # DeepSeek API settings
    DEEPSEEK_API_KEY: str = os.getenv("DEEPSEEK_API_KEY", "dummy")
    DEEPSEEK_MODEL: str = "gpt-5"
    DEEPSEEK_BASE_URL: str = "http://localhost:4141/v1"
    
    # Embedding model settings
    EMBEDDING_MODEL: str = "text-embedding-3-small"
    
    # Document processing settings
    CHUNK_SIZE: int = 800       # Size of text chunks for vector storage
    CHUNK_OVERLAP: int = 100     # Characters to overlap between chunks
    MAX_RESULTS: int = 5         # Maximum search results to return
    MAX_HISTORY: int = 2         # Number of conversation messages to remember
    
    # Database paths
    CHROMA_PATH: str = "./chroma_db"  # ChromaDB storage location

config = Config()


