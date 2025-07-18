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

# Set background image
st.markdown(
    """
    <style>
    .stApp {
        background-image: url("https://saveindia108.in/wp-content/uploads/2025/05/Untitled-design-1-2.png");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Load environment variables
load_dotenv()

# Load the GROQ and Google API Keys
groq_api_key = os.getenv('GROQ_API_KEY')
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

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
    st.session_state.loader = PyPDFDirectoryLoader("./us_census")  # Replace with your PDF directory
    st.session_state.docs = st.session_state.loader.load()
    st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:20])
    st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)

# Input
prompt1 = st.text_input("Enter Your Question From Documents")

# Button to Embed Documents
if st.button("Documents Embedding"):
    vector_embedding()
    st.markdown(
        """
        <div style="background-color: black; padding: 15px; border-radius: 10px;">
            <p style="color: #00FF7F; font-weight: bold; font-size: 16px;">✅ Vector Store DB is ready.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

# Answering the Question
if prompt1:
    if st.session_state.vectors is not None:
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = st.session_state.vectors.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        start = time.process_time()
        response = retrieval_chain.invoke({'input': prompt1})

        # Response time
        st.markdown(
            f"""
            <div style="background-color: black; padding: 10px; border-radius: 10px; margin-top: 10px;">
                <p style="color: #FFD700; font-size: 14px;">🕐 Response time: <b>{round(time.process_time() - start, 2)} seconds</b></p>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Answer block
        st.markdown("### 📝 Answer:")
        st.markdown(
            f"""
            <div style="background-color: black; padding: 15px; border-radius: 10px;">
                <p style="color: white; font-size: 16px;">{response['answer']}</p>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Document Similarity Search (fixed color)
        with st.expander("📄 Document Similarity Search"):
            for i, doc in enumerate(response["context"]):
                st.markdown(
                    f"""
                    <div style="background-color: black; padding: 15px; border-radius: 10px; margin-bottom: 10px;">
                        <p style="color: white; font-size: 14px;">{doc.page_content}</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
    else:
        st.error("❌ Please embed the documents first by clicking 'Documents Embedding'.")
