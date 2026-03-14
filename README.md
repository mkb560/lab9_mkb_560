# lab9_mkb_560

## Overview

This project is a Retrieval-Augmented Generation (RAG) Question-and-Answer chatbot capable of analyzing uploaded PDF documents and answering user queries based on the extracted text. The system provides two functionally equivalent tracks: a cloud-based pipeline using OpenAI's models, and a fully local, open-source pipeline using Meta's Llama 3 and Hugging Face embeddings. It features a user-friendly Flask web interface for automated document uploading and isolated conversational memory.

---

## Deployment

### 1. Environment Setup

Install the required dependencies in your Python environment:

```bash
pip install requirements.txt
```

> **Note:** An OpenMP library conflict on macOS is handled natively in the code by overriding the `KMP_DUPLICATE_LIB_OK` environment variable.

### 2. Model Configuration

- **Optioin 1 (OpenAI):** Requires an active OpenAI API key. Save it in a `.env` file or export it as `OPENAI_API_KEY`.
- **Option 2 (Open-Source):** Requires the open-source [Ollama](https://ollama.com) framework. Install Ollama locally and pull the Llama 3 model by running:

```bash
ollama run llama3
```

### 3. Execution

- **Web Interface (Recommended):** Run `python app.py` and open `http://127.0.0.1:5000` in your browser to access the High-Contrast UI.
- **Command Line (CLI):** Run `python app.py` to test the OpenAI track, or `python app_opensource.py` to test the Open-Source track in your terminal.

---

## Code Description

| File | Description |
|------|-------------|
| `pdf_extractor.py` | Iterates through uploaded PDFs, extracts and cleans text using PyPDF2, splits it into 500-character chunks, and saves the data to a SQLite database (`pdf_data.db`). |
| `vectorstore_builder.py` | Converts text chunks into vector embeddings (using either OpenAI or `BAAI/bge-small-en-v1.5`) and persists them to local FAISS indices. |
| `conversation_chain.py` | Acts as the core RAG engine by connecting the FAISS vector store to the selected LLM and managing multi-turn chat history via `ConversationBufferMemory`. |
| `app.py` | The Flask backend API that manages user sessions and bridges the frontend UI with the Python extraction and conversation logic via `/upload` and `/ask` routes. |
| `index.html`, `style.css`, `script.js` | Powers the Single Page Application (SPA), handling asynchronous file uploads, dynamic chat window updates, and the high-contrast black-and-white UI. |
