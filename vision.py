# Q&A Chatbot using Gemini and Image Upload

from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env

import streamlit as st
import os
from PIL import Image
import google.generativeai as genai

# Configure Gemini API Key
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Function to get response from Gemini
def get_gemini_response(input, image):
    model = genai.GenerativeModel('gemini-1.5-flash')
    if input != "":
        response = model.generate_content([input, image])
    else:
        response = model.generate_content(image)
    return response.text

# Set Streamlit page config
st.set_page_config(page_title="Freedom Guide")

# Add background image, custom font, and label styling
page_style = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins&display=swap');

html, body, [class*="css"]  {
    font-family: 'Poppins', sans-serif;
    color: black !important;
}

/* Background image */
[data-testid="stAppViewContainer"] {
    background-image: url("https://thumbs.dreamstime.com/b/watercolor-background-india-republic-day-celebration-indian-flag-fighter-jets-formation-show-national-tricolor-banner-367157745.jpg");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}

/* Remove default header background */
[data-testid="stHeader"] {
    background-color: rgba(0,0,0,0);
}

/* Black title */
.black-title {
    font-size: 2.5rem;
    font-weight: 700;
    color: black;
    margin-bottom: 10px;
}

/* Force black label text */
label, .stTextInput label, .stFileUploader label,
.st-bb, .st-c6, .st-cg, .st-cb, .st-ch {
    color: black !important;
    font-weight: 600;
}
</style>
"""
st.markdown(page_style, unsafe_allow_html=True)

# Custom black header
st.markdown('<h1 class="black-title"></h1>', unsafe_allow_html=True)

# Input prompt
input = st.text_input("Input Prompt:", key="input")

# Image uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
image = ""

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image.", use_column_width=True)

# Submit button
submit = st.button("Tell me about the image")

# If submit button clicked
if submit:
    response = get_gemini_response(input, image)

    # Display Gemini response in styled box
    st.markdown(
        f"""
        <div style="background-color: black; color: white; padding: 15px; border-radius: 10px;
                    font-family: 'Poppins', sans-serif; margin-top: 20px;">
            <h4>The Response is:</h4>
            <p>{response}</p>
        </div>
        """,
        unsafe_allow_html=True
    )
