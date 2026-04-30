# 📄 DocuMind — AI-Powered PDF Question Answering System

> Upload any PDF. Ask anything. Get accurate, source-grounded answers powered by RAG + LLM.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square&logo=python)
![LangChain](https://img.shields.io/badge/LangChain-0.1%2B-green?style=flat-square)
![FAISS](https://img.shields.io/badge/FAISS-Vector%20DB-orange?style=flat-square)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-red?style=flat-square)
![OpenAI](https://img.shields.io/badge/OpenAI-GPT--3.5%2F4-purple?style=flat-square)

---

## 🧠 What is DocuMind?

**DocuMind** is an end-to-end **Retrieval-Augmented Generation (RAG)** application that lets users upload PDF documents and ask natural language questions about their content. Instead of sending the entire document to an LLM, DocuMind intelligently retrieves only the most relevant chunks and feeds them to the model — resulting in faster, cheaper, and more accurate answers.

---

## 🚀 Demo

```
Upload a PDF → Ask a question → Get an accurate, cited answer in seconds
```

---

## ✨ Features

- 📤 **PDF Upload** — Drag and drop any PDF via the Streamlit UI
- 🔍 **Smart Text Chunking** — Recursive character-based splitting for optimal context windows
- 🧬 **Vector Embeddings** — Converts text chunks to dense vectors using OpenAI Embeddings
- ⚡ **FAISS Vector Store** — Lightning-fast similarity search across thousands of chunks
- 🤖 **LLM Answer Generation** — GPT-3.5 / GPT-4 synthesizes retrieved chunks into precise answers
- 📌 **Source Display** — Shows which document chunks were used to generate the answer
- 💬 **Conversational Memory** — Maintains context across follow-up questions

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                      STREAMLIT UI                        │
│              (Upload PDF + Ask Questions)                │
└────────────────────────┬────────────────────────────────┘
                         │
          ┌──────────────▼──────────────┐
          │        PDF INGESTION         │
          │   PyPDF2 / pdfplumber        │
          └──────────────┬──────────────┘
                         │
          ┌──────────────▼──────────────┐
          │        TEXT CHUNKING         │
          │  RecursiveCharacterSplitter  │
          │  (chunk_size=1000, overlap=200)│
          └──────────────┬──────────────┘
                         │
          ┌──────────────▼──────────────┐
          │      EMBEDDING GENERATION    │
          │   OpenAI text-embedding-ada  │
          └──────────────┬──────────────┘
                         │
          ┌──────────────▼──────────────┐
          │       FAISS VECTOR STORE     │
          │  (index.faiss + index.pkl)   │
          └──────────────┬──────────────┘
                         │
          ┌──────────────▼──────────────┐
          │      SIMILARITY SEARCH       │
          │   Top-K Relevant Chunks      │
          └──────────────┬──────────────┘
                         │
          ┌──────────────▼──────────────┐
          │     LLM ANSWER GENERATION    │
          │   LangChain + GPT-3.5/4      │
          └──────────────┬──────────────┘
                         │
          ┌──────────────▼──────────────┐
          │     DISPLAY ANSWER + SOURCES │
          └─────────────────────────────┘
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| **UI** | Streamlit |
| **PDF Parsing** | PyPDF2, pdfplumber |
| **Text Splitting** | LangChain RecursiveCharacterTextSplitter |
| **Embeddings** | OpenAI `text-embedding-ada-002` |
| **Vector Store** | FAISS (Facebook AI Similarity Search) |
| **LLM** | OpenAI GPT-3.5-turbo / GPT-4 |
| **Orchestration** | LangChain |
| **Language** | Python 3.10+ |

---

## 📁 Project Structure

```
DocuMind/
│
├── app.py                  # Main Streamlit application
├── main.py                 # Core RAG pipeline logic
├── ap.py                   # Helper / utility functions
│
├── requirements.txt        # Python dependencies
│
├── index.faiss             # FAISS vector index (generated at runtime)
├── index.pkl               # Chunk metadata store (generated at runtime)
│
├── .devcontainer/          # Dev container config (for GitHub Codespaces)
│
└── README.md               # You are here
```

---

## ⚙️ Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/m-akanksha49/LLM.git
cd LLM
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Set Your OpenAI API Key

Create a `.env` file in the root directory:

```env
OPENAI_API_KEY=your_openai_api_key_here
```

Or export it directly in your terminal:

```bash
export OPENAI_API_KEY="your_openai_api_key_here"
```

### 5. Run the Application

```bash
streamlit run app.py
```

The app will open automatically at `http://localhost:8501`

---

## 🧪 How It Works — Step by Step

### Step 1 — Upload PDF
User uploads a PDF file via the Streamlit sidebar. The file is read and raw text is extracted page by page using `PyPDF2`.

### Step 2 — Text Chunking
The extracted text is split into overlapping chunks using LangChain's `RecursiveCharacterTextSplitter`:
```python
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
chunks = text_splitter.split_text(raw_text)
```

### Step 3 — Embedding Generation
Each chunk is converted into a high-dimensional vector using OpenAI's embedding model:
```python
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
```

### Step 4 — FAISS Indexing
All chunk vectors are stored in a FAISS index for fast retrieval:
```python
vector_store = FAISS.from_texts(chunks, embeddings)
vector_store.save_local("index")
```

### Step 5 — Similarity Search
When a user submits a question, it is embedded and compared against all chunk vectors. The top-K most similar chunks are retrieved:
```python
docs = vector_store.similarity_search(query, k=4)
```

### Step 6 — LLM Answer Generation
Retrieved chunks are passed as context to the LLM via LangChain's QA chain:
```python
chain = load_qa_chain(llm=ChatOpenAI(), chain_type="stuff")
answer = chain.run(input_documents=docs, question=query)
```

### Step 7 — Display Results
The answer and source chunks are rendered in the Streamlit UI.

---

## 📦 Requirements

```txt
streamlit
langchain
langchain-openai
faiss-cpu
openai
PyPDF2
pdfplumber
python-dotenv
tiktoken
```

Install all at once:
```bash
pip install -r requirements.txt
```

---

## 🔧 Configuration

You can tune the following parameters in `main.py` to optimize performance:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `chunk_size` | `1000` | Number of characters per chunk |
| `chunk_overlap` | `200` | Overlap between consecutive chunks |
| `k` (retrieval) | `4` | Number of chunks retrieved per query |
| `model_name` | `gpt-3.5-turbo` | LLM model used for answer generation |
| `temperature` | `0` | LLM temperature (0 = deterministic) |

---

## 🧯 Common Issues & Fixes

| Issue | Fix |
|-------|-----|
| `AuthenticationError` | Check your `OPENAI_API_KEY` in `.env` |
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` again |
| PDF not parsing | Try `pdfplumber` instead of `PyPDF2` for scanned PDFs |
| Empty answers | Increase `k` value or reduce `chunk_size` |
| Slow responses | Switch to `gpt-3.5-turbo` for faster inference |

---

## 🔮 Future Improvements

- [ ] Support for multiple PDF uploads simultaneously
- [ ] Add HuggingFace open-source embeddings (sentence-transformers) as a free alternative
- [ ] Implement re-ranking with Cohere for better retrieval accuracy
- [ ] Add chat history with memory buffer
- [ ] Support for DOCX, TXT, and CSV file formats
- [ ] Deploy on Streamlit Cloud / HuggingFace Spaces

---

## 👩‍💻 Author

**Akanksha M**
- GitHub: [@m-akanksha49](https://github.com/m-akanksha49)
- Project Repository: [github.com/m-akanksha49/LLM](https://github.com/m-akanksha49/LLM)

---

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).

---

> ⭐ If you found this project helpful, please consider giving it a star on GitHub!
