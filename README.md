# Privacy-Preserving Health AI Assistant

A conversational healthcare information assistant with:

- Context-aware health conversation
- Medical report upload
- Report question answering
- RAG-based evidence retrieval
- PII redaction
- Evidence verification
- Gemini-powered generation
- Trained conversation router

## Local Run

Install dependencies:

    pip install -r requirements.txt

Run:

    streamlit run app.py

## Required Secret

Configure GEMINI_API_KEY in Streamlit secrets.

Never commit API keys or encryption keys.

## Disclaimer

This application provides health information and report explanation. It is not a diagnostic system and does not replace professional medical care.