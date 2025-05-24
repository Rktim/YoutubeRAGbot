import os
from typing import Optional
from dotenv import load_dotenv
import warnings
import numpy as np

# Load environment variables
load_dotenv()

# Suppress warnings
warnings.filterwarnings('ignore')

try:
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_groq import ChatGroq
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import Chroma
    from langchain.chains import ConversationalRetrievalChain
    from langchain.memory import ConversationBufferMemory
except ImportError as e:
    print(f"Error importing required packages: {e}")
    print("Please install all required packages using: pip install -r requirements.txt")
    raise

from get_transcripts import get_video_transcript

class YouTubeRAG:
    def __init__(self):
        """Initialize the YouTube RAG system."""
        try:
            # Get Groq API key from environment
            groq_api_key = os.getenv("GROQ_API_KEY")
            if not groq_api_key:
                raise ValueError("GROQ_API_KEY not found in environment variables")

            # Initialize embeddings with specific model and settings
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
            
            # Initialize vector store
            self.vector_store = None
            
            # Initialize chat model
            self.chat_model = ChatGroq(
                api_key=groq_api_key,
                model_name="mixtral-8x7b-32768"
            )
            
            # Initialize memory
            self.memory = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True
            )
            
            # Initialize text splitter
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            
        except Exception as e:
            print(f"Error initializing YouTubeRAG: {e}")
            raise

    def load_video(self, youtube_link: str) -> bool:
        """Load and process a YouTube video transcript."""
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
            self.vector_store = Chroma.from_texts(
                chunks,
                self.embeddings,
                collection_name="youtube_transcript",
                persist_directory="db"
            )

            return True
        except Exception as e:
            print(f"Error loading video: {e}")
            return False

    def chat(self, query: str) -> str:
        """Process a chat query and return a response."""
        try:
            if not self.vector_store:
                return "Please load a video first using the 'Get Transcript' button."
            
            # Create the chain
            chain = ConversationalRetrievalChain.from_llm(
                llm=self.chat_model,
                retriever=self.vector_store.as_retriever(),
                memory=self.memory
            )
            
            # Get response
            response = chain({"question": query})
            return response["answer"]
            
        except Exception as e:
            print(f"Error in chat: {e}")
            return f"An error occurred: {str(e)}"

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
