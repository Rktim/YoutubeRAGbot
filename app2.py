import gradio as gr
import os
import re
from youtube_rag import YouTubeRAG
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize RAG system
rag = YouTubeRAG()

def extract_video_id(youtube_link: str) -> str:
    """Extract video ID from a YouTube URL."""
    video_id_pattern = r'(?:https?:\/\/)?(?:www\.)?(?:youtube\.com\/(?:[^\/\n\s]+\/\S+\/|(?:v|e(?:mbed)?)\/|\S*?[?&]v=)|youtu\.be\/)([a-zA-Z0-9_-]{11})'
    match = re.match(video_id_pattern, youtube_link)
    if match:
        return match.group(1)
    return None

def process_video(youtube_link: str, history):
    """Process the YouTube video and return a message."""
    if not youtube_link:
        return history + [["", "Please enter a YouTube link."]]
    
    video_id = extract_video_id(youtube_link)
    if not video_id:
        return history + [["", "❌ Invalid YouTube link. Please enter a valid YouTube URL."]]
    
    try:
        # Create embeddings directory
        os.makedirs("db", exist_ok=True)
        
        # Load video
        if rag.load_video(youtube_link):
            return history + [["", "✅ Video transcript loaded! You can now ask questions about the content."]]
        else:
            return history + [["", "❌ Failed to load video. Please check if the video has captions available."]]
    except Exception as e:
        return history + [["", f"❌ Error processing video: {str(e)}"]]

def chat(message, history, youtube_link):
    """Handle chat messages."""
    if not youtube_link:
        return history + [[message, "Please enter a YouTube link first."]]
    
    try:
        response = rag.chat(message)
        return history + [[message, response]]
    except Exception as e:
        return history + [[message, f"❌ Error: {str(e)}"]]

# Custom CSS with YouTube-inspired theme
css = """
.gradio-container {
    background: #0f0f0f;
    color: white;
    font-family: 'Roboto', sans-serif;
}
.container {
    max-width: 1200px;
    margin: 0 auto;
    padding: 2rem;
}
.chat-message {
    padding: 1rem;
    border-radius: 8px;
    margin: 0.5rem 0;
    background: #1f1f1f;
    border: 1px solid #303030;
}
.user-message {
    background: #1f1f1f;
    border-left: 4px solid #ff0000;
}
.assistant-message {
    background: #1f1f1f;
    border-left: 4px solid #3ea6ff;
}
.gradio-button {
    background: #ff0000 !important;
    color: white !important;
    border: none !important;
    border-radius: 2px !important;
    padding: 8px 16px !important;
    font-weight: 500 !important;
    transition: background-color 0.2s !important;
}
.gradio-button:hover {
    background: #cc0000 !important;
}
.gradio-textbox {
    background: #1f1f1f !important;
    border: 1px solid #303030 !important;
    color: white !important;
    border-radius: 2px !important;
}
.gradio-textbox:focus {
    border-color: #ff0000 !important;
}
.gradio-label {
    color: #aaaaaa !important;
    font-size: 0.9em !important;
}
.chatbot {
    background: #1f1f1f !important;
    border: 1px solid #303030 !important;
    border-radius: 8px !important;
}
.chatbot .message {
    padding: 12px !important;
    margin: 8px 0 !important;
    border-radius: 8px !important;
}
.chatbot .user-message {
    background: #1f1f1f !important;
    border-left: 4px solid #ff0000 !important;
}
.chatbot .assistant-message {
    background: #1f1f1f !important;
    border-left: 4px solid #3ea6ff !important;
}
"""

# Create the Gradio interface
with gr.Blocks(css=css) as demo:
    gr.Markdown("""
    <div style='text-align: center; padding: 20px;'>
        <h1 style='color: #ff0000; font-size: 2.5em; margin-bottom: 10px;'>YouTube RAGbot 🎥</h1>
        <p style='color: #aaaaaa; font-size: 1.2em;'>Your AI-powered video analysis companion</p>
    </div>
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            # YouTube link input
            youtube_link = gr.Textbox(
                label="🎬 Enter YouTube Link",
                placeholder="Paste your YouTube link here...",
                lines=1,
                elem_classes=["youtube-input"]
            )
            
            # Video display
            video = gr.HTML()
            
            # Process button
            process_btn = gr.Button("✨ Get Transcript", elem_classes=["youtube-button"])
        
        with gr.Column(scale=1):
            # Chat interface
            chatbot = gr.Chatbot(
                label="💬 Chat",
                height=600,
                show_copy_button=True,
                elem_classes=["youtube-chat"]
            )
            
            # Chat input
            msg = gr.Textbox(
                label="Ask a question about the video...",
                placeholder="Type your question here...",
                lines=2,
                elem_classes=["youtube-input"]
            )
            
            # Submit button
            submit_btn = gr.Button("Send", elem_classes=["youtube-button"])
    
    # Update video display when link is entered
    def update_video(youtube_link):
        video_id = extract_video_id(youtube_link)
        if video_id:
            return f'<div style="background: #000; padding: 20px; border-radius: 8px;"><iframe width="100%" height="315" src="https://www.youtube.com/embed/{video_id}" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe></div>'
        return ""
    
    # Set up event handlers
    youtube_link.change(
        fn=update_video,
        inputs=[youtube_link],
        outputs=[video]
    )
    
    process_btn.click(
        fn=process_video,
        inputs=[youtube_link, chatbot],
        outputs=[chatbot]
    )
    
    submit_btn.click(
        fn=chat,
        inputs=[msg, chatbot, youtube_link],
        outputs=[chatbot]
    ).then(
        fn=lambda: "",
        inputs=[],
        outputs=[msg]
    )
    
    msg.submit(
        fn=chat,
        inputs=[msg, chatbot, youtube_link],
        outputs=[chatbot]
    ).then(
        fn=lambda: "",
        inputs=[],
        outputs=[msg]
    )

# Launch the app
if __name__ == "__main__":
    demo.launch(share=True) 