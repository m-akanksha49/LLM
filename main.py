import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
import time

load_dotenv()

# Load the GROQ and Google API Keys
groq_api_key = os.getenv('GROQ_API_KEY')
os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]

st.title("Voices of Freedom: Ask Our Heroes ")

# Initialize LLM
llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")

# Prompt Template
prompt = ChatPromptTemplate.from_template("""
Answer the questions and explain fully about the Indias Related information only based on the given context.
Please provide the accurate response based on the question.

<context>
{context}
<context>
Questions: {input}
""")

# Initialize session state
if "vectors" not in st.session_state:
    st.session_state.vectors = None

def vector_embedding():
    st.session_state.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    st.session_state.loader = PyPDFDirectoryLoader("./us_census")  # Data Ingestion
    st.session_state.docs = st.session_state.loader.load()  # Document Loading
    st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)  # Chunk Creation
    st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:20])  # Splitting
    st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)  # Vector DB

# UI Input
prompt1 = st.text_input("Enter Your Question From Documents")

# Button to Embed Documents
if st.button("Documents Embedding"):
    vector_embedding()
    st.success("✅ Vector Store DB is ready.")

# Answering the Question
if prompt1:
    if st.session_state.vectors is not None:
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = st.session_state.vectors.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        start = time.process_time()
        response = retrieval_chain.invoke({'input': prompt1})
        st.write("🕐 Response time:", round(time.process_time() - start, 2), "seconds")
        st.write("### 📝 Answer:")
        st.write(response['answer'])

        with st.expander("📄 Document Similarity Search"):
            for i, doc in enumerate(response["context"]):
                st.write(doc.page_content)
                st.write("---")
    else:
        st.error("❌ Please embed the documents first by clicking 'Documents Embedding'.")
