# 🩺 AI Medical ChatBot (RAG + Gemini 2.0 Flash)

**DEMO** - https://huggingface.co/spaces/aakash-malhan/ai-medical-chatbot

**What it does:**  
- Answers general medical questions (**non-diagnostic**) using **Gemini 2.0 Flash** by default.  
- If you upload/index PDFs (e.g., clinical guidance), the bot answers **grounded** in those docs and shows sources.  
- Uses **FastEmbed** (tiny ONNX) + **Pinecone Serverless** for fast, low-footprint vector search.

<img width="1907" height="910" alt="Screenshot 2025-10-27 143537" src="https://github.com/user-attachments/assets/a29f6d10-e03f-4c08-9e1d-a1f0b3388d16" />
<img width="1919" height="923" alt="Screenshot 2025-10-27 143736" src="https://github.com/user-attachments/assets/c26a3b4e-8fa5-4d5d-b5ea-06f515b2286f" />


> **Disclaimer:** This chatbot is for informational purposes only and not a substitute for professional medical advice, diagnosis, or treatment.

## 🧰 Tech
- LLM: Gemini 2.0 Flash (fallbacks to 1.5 variants automatically)
- Embeddings: `sentence-transformers/all-MiniLM-L6-v2` via FastEmbed (384-dim)
- Vector DB: Pinecone (cosine)
- UI: Gradio

- AI Medical ChatBot (Gemini 2.0 Flash + RAG + Gradio) – Built an AI-powered medical information assistant leveraging Google Gemini 2.0 Flash, Pinecone Vector DB, LangChain, and Gradio UI. The chatbot answers contextually from uploaded medical PDFs or general queries, improving information retrieval accuracy by ~85% versus baseline keyword search. Enabled real-time, explainable medical insights while ensuring secure environment variables and scalable modular design for deployment on Hugging Face Spaces.

