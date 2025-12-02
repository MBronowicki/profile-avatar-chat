---
title: Your App Name
emoji: 🤖
sdk: gradio
app_file: app.py
pinned: false
---


# Profile Avatar Chat App

This repository contains the code for a robust AI-powered chat service that acts as a personal profile avatar. The chat responds based on my LinkedIn profile, professional and other additional information.

Key features implemented for robustness:

 - Semantic QA cache: Reuses previous answers for repeated or similar questions to improve response speed and consistency.

- Embedding-based similarity search: Uses OpenAI embeddings and cosine similarity to find semantically similar past questions and refine answers.

- Sliding window conversation context: Keeps only the last n messages for token-efficient API calls while preserving relevant context.

- Automated evaluation and rerun: Uses Google Gemini (via OpenAI API wrapper) to evaluate generated responses, automatically rerunning and refining answers when quality control flags them.

- Traceability with LangSmith: Key functions are decorated for run tracking, enabling debugging and historical inspection of chat interactions.

- PDF and text ingestion: Extracts profile information from LinkedIn PDF, summary, current situation, and recommendation text files.

- Gradio integration: Provides an interactive chat interface for local testing and deployment.

This chat service powers my portfolio website, which communicates with this deployed Hugging Face Space for live interactions.
