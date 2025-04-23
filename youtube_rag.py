from typing import Optional
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
import os
import warnings
import numpy as np
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Filter out numpy deprecation warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

from get_transcripts import get_video_transcript

class youtuberag:
    def __init__(self, model_name: str = "llama3-70b-8192"):
        """Initialize the RAG system with specified LLM model."""
        try:
            # Create a persistent directory for embeddings
            os.makedirs("db", exist_ok=True)

            # Get Groq API key from environment
            groq_api_key = os.getenv("GROQ_API_KEY")
            if not groq_api_key:
                raise ValueError("GROQ_API_KEY not found in environment variables")

            # Load SentenceTransformer model from local directory
            self.embeddings = HuggingFaceEmbeddings(
                model_name="local_miniLM_model",  # Local folder instead of Hugging Face hub
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )

            # Initialize LLM
            self.llm = ChatGroq(
                model_name=model_name,
                temperature=0.7,
                groq_api_key=groq_api_key
            )

            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=50,
                length_function=len
            )
            self.memory = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True,
                output_key="answer"
            )
            self.chain = None
            self.db = None
        except Exception as e:
            print(f"Error initializing RAG system: {e}")
            raise

    def load_video(self, youtube_link: str) -> bool:
        """Load and process a YouTube video transcript into the RAG system."""
        try:
            transcript = get_video_transcript(youtube_link)
            if not transcript:
                print("Could not fetch transcript from the video")
                return False

            chunks = self
