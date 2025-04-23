import streamlit as st
from youtube_rag import youtuberag
import os
import re
import time

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
        color: #FF0000  !important;
        max-width: 800px;
    }
    .chat-message {
        padding: 1rem;
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
    h1, h2, h3, p {
        color: #ffffff;
    }
    .stMarkdown {
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

if "rag" not in st.session_state:
    st.session_state.rag = YouTubeRAG()

if "current_video_id" not in st.session_state:
    st.session_state.current_video_id = None

if "processing" not in st.session_state:
    st.session_state.processing = False

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
    youtube_link = st.text_input("", key="youtube_link", placeholder="Paste your YouTube link here...")
    
    if youtube_link:
        video_id = extract_video_id(youtube_link)
        if video_id:
            st.session_state.current_video_id = video_id
            st.video(f"https://www.youtube.com/watch?v={video_id}")
            if st.button("✨ Get Transcript", key="process_button"):
                st.session_state.processing = True
                with st.spinner("🔍 Processing video transcript..."):
                    # Create embeddings directory
                    os.makedirs("db", exist_ok=True)
                    
                    # Load video
                    if st.session_state.rag.load_video(youtube_link):
                        st.success("✅ Video transcript loaded! You can now ask questions about the content.")
                        # Clear previous chat history when loading new video
                        st.session_state.messages = []
                    else:
                        st.error("❌ Failed to load video. Please check if the video has captions available.")
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
    if prompt := st.chat_input("Ask a question about the video...", key="chat_input"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("🤔 Thinking..."):
                response = st.session_state.rag.chat(prompt)
                st.markdown(response)
                
                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": response})

