# 🧠 DocMind — AI-Powered PDF Intelligence (RAG System)

DocMind is an advanced **Retrieval-Augmented Generation (RAG)** application that allows users to upload PDF documents and interact with them using natural language queries. It combines **semantic search, vector embeddings, and Large Language Models (LLMs)** to deliver accurate, context-aware, and source-grounded answers.

---

## 🚀 Features

✨ Upload and analyze multiple PDF documents  
✨ Intelligent **semantic search** using vector embeddings  
✨ Context-aware **question answering using LLMs**  
✨ **FAISS-powered vector indexing** for fast retrieval  
✨ Optimized **chunking strategies** for better accuracy  
✨ Reduces hallucinations with **source-grounded responses**  
✨ Interactive UI built with **Streamlit**  

---

## 🏗️ Architecture Overview 


User Query
↓
Streamlit UI
↓
PDF Upload & Text Extraction
↓
Text Chunking
↓
Embedding Generation (OpenAI)
↓
FAISS Vector Database
↓
Similarity Search
↓
Relevant Context Retrieved
↓
LLM (OpenAI) → Answer Generation
↓
Response Displayed to User
