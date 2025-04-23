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

class YouTubeRAG:
    def __init__(self, model_name: str = "qwen-qwq-32b"):
        """Initialize the RAG system with specified LLM model."""
        try:
            # Create a persistent directory for embeddings
            os.makedirs("db", exist_ok=True)
            
            
            # Get Groq API key from environment
            groq_api_key = os.getenv("GROQ_API_KEY")
            if not groq_api_key:
                raise ValueError("GROQ_API_KEY not found in environment variables")
            
            self.embeddings = HuggingFaceEmbeddings(
                model_name="all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
            
            # Initialize Groq LLM with error handling
            try:
                self.llm = ChatGroq(
                    model_name=model_name,
                    temperature=0.7,
                    groq_api_key=groq_api_key
                )
            except Exception as e:
                raise ValueError(f"Failed to initialize Groq LLM: {str(e)}")
            
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
            # Get transcript using our transcript fetcher
            transcript = get_video_transcript(youtube_link)
            if not transcript:
                print("Could not fetch transcript from the video")
                return False

            # Split transcript into chunks
            chunks = self.text_splitter.split_text(transcript)
            if not chunks:
                print("Could not split transcript into chunks")
                return False

            # Create vector store
            print("Creating vector store...")
            self.db = Chroma.from_texts(
                chunks,
                self.embeddings,
                collection_name="youtube_transcript",
                persist_directory="db"
            )

            # Create conversation chain
            print("Setting up conversation chain...")
            self.chain = ConversationalRetrievalChain.from_llm(
                llm=self.llm,
                retriever=self.db.as_retriever(search_kwargs={"k": 3}),
                memory=self.memory,
                return_source_documents=True,
                output_key="answer"
            )

            return True
        except Exception as e:
            print(f"Error loading video: {str(e)}")
            return False

    def chat(self, query: str) -> Optional[str]:
        """Chat with the RAG system about the video content."""
        if not self.chain:
            return "Please load a video first using load_video(youtube_link)"

        try:
            result = self.chain({"question": query})
            return result['answer']
        except Exception as e:
            return f"An error occurred during chat: {str(e)}"

def main():
    # Initialize the RAG system
    print("Initializing RAG system...")
    try:
        rag = YouTubeRAG()
    except Exception as e:
        print(f"Failed to initialize RAG system: {e}")
        return
    
    # Get YouTube link from user
    youtube_link = input("Enter a YouTube link: ")
    
    # Load and process the video
    print("\nProcessing video transcript...")
    if not rag.load_video(youtube_link):
        print("Failed to load video. Exiting...")
        return

    print("\nVideo transcript loaded! You can now ask questions about the video content.")
    print("Type 'quit' to exit")

    # Chat loop
    while True:
        try:
            user_input = input("\nYou: ")
            if user_input.lower() == 'quit':
                break

            response = rag.chat(user_input)
            print(f"\nAssistant: {response}")
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"\nAn error occurred: {e}")
            print("You can continue chatting or type 'quit' to exit")

if __name__ == "__main__":
    main() 
