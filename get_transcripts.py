import re
from youtube_transcript_api import YouTubeTranscriptApi
from typing import Optional, Dict, List

def extract_video_id(youtube_link: str) -> Optional[str]:
    """Extract video ID from a YouTube URL."""
    video_id_pattern = r'(?:https?:\/\/)?(?:www\.)?(?:youtube\.com\/(?:[^\/\n\s]+\/\S+\/|(?:v|e(?:mbed)?)\/|\S*?[?&]v=)|youtu\.be\/)([a-zA-Z0-9_-]{11})'
    match = re.match(video_id_pattern, youtube_link)
    if match:
        return match.group(1)
    return None

def get_transcript(video_id: str) -> Optional[str]:
    """Get transcript text from a YouTube video."""
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        text = " ".join([entry['text'] for entry in transcript])
        return text
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def get_video_transcript(youtube_link: str) -> Optional[str]:
    """Main function to get transcript from a YouTube link."""
    video_id = extract_video_id(youtube_link)
    if not video_id:
        print("Invalid YouTube link")
        return None
    
    return get_transcript(video_id)

if __name__ == "__main__":
    # Example usage
    youtube_link = input("Enter a YouTube link: ")
    transcript = get_video_transcript(youtube_link)
    if transcript:
        print("Transcript retrieved successfully!")
        print("-" * 50)
        print(transcript)