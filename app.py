from dotenv import load_dotenv
import streamlit as st
import os
import google.generativeai as genai

# Load environment variables
load_dotenv()

# Configure Google Gemini API
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Initialize the Generative Model
model = genai.GenerativeModel("gemini-pro")
chat = model.start_chat(history=[])

# Function to get a response from Google Gemini
def get_gemini_response(question):
    response = chat.send_message(question, stream=True)
    return response

# Streamlit App Configuration
st.set_page_config(page_title="PulsePlay", layout="centered")

st.header("Binod Chatbot")

# Chat history management
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

# Input and submit functionality
user_input = st.text_input("Ask your question:")
if st.button("Submit") and user_input:
    # Append user query to chat history
    st.session_state['chat_history'].append(("You", user_input))
    
    # Get the Gemini response
    response = get_gemini_response(user_input)
    
    # Display the response and append to chat history
    st.subheader("Response:")
    for chunk in response:
        st.write(chunk.text)
        st.session_state['chat_history'].append(("PulsePlay", chunk.text))

# Display chat history
st.subheader("Chat History:")
for role, text in st.session_state['chat_history']:
    st.write(f"{role}: {text}")
