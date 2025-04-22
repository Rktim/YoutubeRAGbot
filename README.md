
# 🎬 YouTube RAGbot

*Your AI-powered video analysis companion—now with a snazzy web UI!
Dive into any YouTube video, pull out its transcript, and chat with an AI that “knows” exactly what’s happening on screen. Perfect for researchers, students, or anyone who wants answers without endless scrubbing.!*

![image](https://github.com/user-attachments/assets/45aeba24-0898-4389-9f34-448813eaceb9)


---

## ✨ Highlights

- **⏩ Instant Video Embed**  
  Paste any YouTube URL and watch it right in-app.

- **📜 One‑Click Transcript**  
  Hit **Get Transcript** to fetch & chunk captions behind the scenes.

- **💬 Live Chat Q&A**  
  Ask anything—“What is RAG?”, “How does it work?”—and watch the bot reply in style.

- **🎨 Dark‑Mode Chic**  
  Eye‑friendly navy theme with vibrant input outlines and chat bubbles.


## 🛠️ Under the Hood

1. **YouTube Embedder**  
   - Parses URL, auto‑embeds player.
2. **Transcript Fetcher**  
   - Leverages `youtube_transcript_api`.
3. **Chunk + Embed**  
   - Splits text ➡️ generates embeddings via OpenAI/Anthropic.
4. **Vector Store**  
   - Uses FAISS for lightning‑fast retrieval.
5. **Chat UI**  
   - Powered by Streamlit’s components + `rich`‑style CSS.

---

## 🎉 Features

- Multi‑LLM compatible (OpenAI, Anthropic)  
- Adjustable chunk/​retrieve params in sidebar  
- Clean, interactive Streamlit UI  
- Chat history with colorful bubbles  
- Mobile‑responsive layout  

---


## 💡 Contribute

1. Fork & branch  
2. Code your magic  
3. Open a PR  
4. Bask in ⭐ glory  

---

## 📜 License

MIT © 2025 Rktim

---

*Ready to dive deep into YouTube videos? Paste a link, hit play, and start chatting!* 🎙️
