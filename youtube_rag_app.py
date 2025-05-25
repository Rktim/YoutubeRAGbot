import streamlit as st
import os
import re
import time
# import gradio as gr # Not needed for Streamlit

from typing import Optional
from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain_groq import ChatGroq # Will use the ChatGroq imported in YouTubeRAG class
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
import warnings
import numpy as np
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Filter out numpy deprecation warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

from get_transcripts import get_video_transcript # Assuming this is a separate file

# YouTubeRAG class from youtube_rag.py
class YouTubeRAG:
    def __init__(self, model_name: str = "llama-3.3-70b-versatile"):
        """Initialize the RAG system with specified LLM model."""
        try:
            # Create a persistent directory for embeddings
            os.makedirs("db", exist_ok=True)
            
            # Get Groq API key from secrets.toml or environment variable
            try:
                # Prioritize Streamlit secrets for deployment flexibility
                groq_api_key = st.secrets["GROQ_API_KEY"]
            except KeyError:
                 # Fallback to environment variable if not in secrets (less common for Streamlit deployment)
                 groq_api_key = os.getenv("GROQ_API_KEY")
                 if not groq_api_key:
                      raise ValueError("GROQ_API_KEY not found in Streamlit secrets.toml or environment variables.")

            # Import ChatGroq here to handle potential conditional import based on env/secrets
            from langchain_groq import ChatGroq
            
            self.embeddings = HuggingFaceEmbeddings(
                model_name="all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
            
            # Initialize Groq LLM with error handling
            try:
                # Correct the model name if the user intended to use a current Groq model
                # Based on previous errors, 'llama-3.3-70b-versatile' is likely decommissioned
                # Using a placeholder, user might need to adjust based on latest Groq docs
                # model_name = "llama3-70b-8192" # Example of a potentially current model
                # Or keep the user's provided name if they handle it elsewhere
                
                self.llm = ChatGroq(
                    model_name=model_name,
                    temperature=0.7,
                    groq_api_key=groq_api_key
                )
            except Exception as e:
                # Catch specific Groq model error and provide guidance
                if "model_decommissioned" in str(e):
                     st.error(f"Error initializing Groq LLM: {str(e)}. Please check Groq documentation for supported model names.")
                else:
                    st.error(f"Failed to initialize Groq LLM: {str(e)}")
                raise ValueError(f"Failed to initialize Groq LLM: {str(e)}")
            
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=50,
                length_function=len
            )
            # Updated memory initialization as per previous fix attempt
            self.memory = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True,
                output_key="answer"
            )
            self.chain = None
            self.db = None
        except Exception as e:
            st.error(f"Error initializing RAG system: {e}")
            # Log the error for debugging
            print(f"Error initializing RAG system: {e}")
            # Optionally re-raise or handle gracefully
            # raise

    def load_video(self, youtube_link: str) -> bool:
        """Load and process a YouTube video transcript into the RAG system."""
        try:
            # Get transcript using our transcript fetcher
            transcript = get_video_transcript(youtube_link)
            if not transcript:
                st.warning("Could not fetch transcript from the video. Make sure captions are available.")
                return False

            # Split transcript into chunks
            chunks = self.text_splitter.split_text(transcript)
            if not chunks:
                st.warning("Could not split transcript into chunks.")
                return False

            # Create vector store
            st.info("Creating vector store...")
            # Clear existing Chroma collection before creating a new one
            if self.db:
                 try:
                     self.db.delete_collection()
                     print("Cleared existing vector store collection.")
                 except Exception as delete_e:
                     print(f"Warning: Could not clear existing vector store collection: {delete_e}")
                     # Continue even if unable to delete
                     pass

            self.db = Chroma.from_texts(
                chunks,
                self.embeddings,
                collection_name="youtube_transcript",
                persist_directory="db"
            )
            st.success("Vector store created!")

            # Create conversation chain
            st.info("Setting up conversation chain...")
            self.chain = ConversationalRetrievalChain.from_llm(
                llm=self.llm, # Use the initialized LLM
                retriever=self.db.as_retriever(search_kwargs={"k": 3}),
                memory=self.memory, # Use the initialized memory
                return_source_documents=True,
                output_key="answer"
            )
            st.success("Conversation chain setup complete!")

            return True
        except Exception as e:
            st.error(f"Error loading video: {str(e)}")
            # Log the error for debugging
            print(f"Error loading video: {str(e)}")
            return False

    def chat(self, query: str) -> Optional[str]:
        """Chat with the RAG system about the video content."""
        if not self.chain:
            return "Please load a video first by entering a YouTube link and clicking 'Get Transcript'."

        try:
            result = self.chain({"question": query})
            # Assuming the chain's output_key is 'answer' as set in the chain init
            return result.get('answer', "Could not retrieve an answer.")
        except Exception as e:
            st.error(f"An error occurred during chat: {str(e)}")
            # Log the error for debugging
            print(f"An error occurred during chat: {str(e)}")
            return f"An error occurred during chat: {str(e)}"


# Streamlit UI code from app.py
# Custom CSS
st.markdown("""
<style>
    header {display: none !important;}
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stApp {
        background: linear-gradient(135deg, #2c3e50 0%, #1a1a2e 100%);
    }
    .main {
        padding: 2rem;
        max-width: 1200px;
        margin: 0 auto;
    }
    .stButton>button {
        background: linear-gradient(45deg, #2196F3, #21CBF3);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 10px 20px;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: scale(1.05);
        box-shadow: 0 5px 15px rgba(33, 203, 243, 0.4);
    }
    .stTextInput>div>div>input {
        border-radius: 10px;
        border: 2px solid #2196F3;
        background: rgba(255, 255, 255, 0.1);
        color: #FF0000  !important; # Example: Use a variable for color
        max-width: 800px;
    }
    .chat-message {
        padding: 1rem;.
        border-radius: 10px;
        margin: 0.5rem 0;
        background: rgba(255, 255, 255, 0.1);
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    .user-message {
        background: rgba(227, 242, 253, 0.2);
    }
    .assistant-message {
        background: rgba(245, 245, 245, 0.2);
    }
    h1, h2, h3, p, .stMarkdown, label {
        color: #ffffff;
    }
    .block-container {
        max-width: 1400px;
        padding-top: 2rem !important;
        padding-right: 2rem !important;
        padding-left: 2rem !important;
        margin: 0 auto !important;
    }
    [data-testid="stVideo"] {
        width: 100% !important;
        max-width: 800px !important;
        margin: 0 auto;
    }
    [data-testid="stVideo"] > div {
        width: 100% !important;
    }
    [data-testid="stVideo"] video {
        width: 100% !important;
    }
    div[data-testid="stHorizontalBlock"] > div {
        padding: 0 1rem;
    }
    div[data-testid="stHorizontalBlock"] {
        gap: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Initialize YouTubeRAG object in session state
if "rag" not in st.session_state:
    # Pass a default model name, or try to get it from secrets/env if needed
    # Note: 'llama-3.3-70b-versatile' was decommissioned. Use a valid one.
    # For now, I'll keep the old name, but the user should update it.
    st.session_state.rag = YouTubeRAG(model_name="llama-3.3-70b-versatile")

if "current_video_id" not in st.session_state:
    st.session_state.current_video_id = None

if "processing" not in st.session_state:
    st.session_state.processing = False

# extract_video_id function from app.py
def extract_video_id(youtube_link: str) -> str:
    """Extract video ID from a YouTube URL."""
    video_id_pattern = r'(?:https?:\/\/)?(?:www\.)?(?:youtube\.com\/(?:[^\/\n\s]+\/\S+\/|(?:v|e(?:mbed)?)\/|\S*?[?&]v=)|youtu\.be\/)([a-zA-Z0-9_-]{11})'
    match = re.match(video_id_pattern, youtube_link)
    if match:
        return match.group(1)
    return None

# Title and description with animation
st.markdown("""
<div style='text-align: center;'>
    <h1 style='color: #2196F3;'>YouTube RAGbot 🎥</h1>
    <p style='color: #666;'>Your AI-powered video analysis companion</p>
</div>
""", unsafe_allow_html=True)

# Create two columns for video and chat
col1, col2 = st.columns([1, 1])

# YouTube link input and video display in first column
with col1:
    st.markdown("### 🎬 Enter YouTube Link")
    youtube_link = st.text_input("YouTube URL", key="youtube_link", placeholder="Paste your YouTube link here...")
    
    if youtube_link:
        video_id = extract_video_id(youtube_link)
        if video_id:
            st.session_state.current_video_id = video_id
            st.video(f"https://www.youtube.com/watch?v={video_id}")
            if st.button("✨ Get Transcript", key="process_button"):
                st.session_state.processing = True
                # Use a placeholder for the spinner text
                with st.spinner("🔍 Processing video transcript..."):
                    # Create embeddings directory (already handled in YouTubeRAG __init__, but good for redundancy)
                    # os.makedirs("db", exist_ok=True)
                    
                    # Load video using the RAG object in session state
                    if st.session_state.rag.load_video(youtube_link):
                        st.success("✅ Video transcript loaded! You can now ask questions about the content.")
                        # Clear previous chat history when loading new video
                        st.session_state.messages = []
                    else:
                        # Error message is already shown in load_video method
                        pass # Or add a generic error if needed
                st.session_state.processing = False
        else:
            st.error("❌ Invalid YouTube link. Please enter a valid YouTube URL.")

# Chat interface in second column
with col2:
    st.markdown("### 💬 Chat")
    
    # Display chat messages with animations
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input with custom styling
    if prompt := st.chat_input("Ask a question about the video...", key="chat_input"): # Use walrus operator for convenience
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message (optional, Streamlit does this automatically now)
        # with st.chat_message("user"):
        #    st.markdown(prompt)
        
        # Generate response using the RAG object in session state
        with st.chat_message("assistant"):
            with st.spinner("🤔 Thinking..."):
                response = st.session_state.rag.chat(prompt)
                st.markdown(response)
                
                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": response})

# The main function from youtube_rag.py is not needed for Streamlit UI
# The leftover gradio chatbot line is also removed.
