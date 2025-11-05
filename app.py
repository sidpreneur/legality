import streamlit as st
import requests
import json

# --- Page Configuration ---
st.set_page_config(
    page_title="Legality",
    page_icon="👮🏽‍♀️",
    layout="wide"
)

# --- App Title ---
st.title("🩺 Law RAG Chatbot")
st.caption("Powered by LangChain, Pinecone, and Hugging Face")

# --- API Configuration ---
# This is the URL where your FastAPI app is running
FASTAPI_BACKEND_URL = "http://127.0.0.1:8000/chat"

# --- Session State for Chat History ---
# We use session state to store the chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I am a virtual law consultant. How can I help you today?"}
    ]

# --- Display Chat History ---
# Loop through the messages in session state and display them
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- Chat Input ---
# This creates the text input box at the bottom of the screen
if prompt := st.chat_input("Ask a question about the law..."):
    
    # 1. Add the user's message to session state and display it
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Prepare the request for the FastAPI backend
    # The payload must match your FastAPI's Pydantic model (`ChatRequest`)
    payload = {"msg": prompt}

    # 3. Call the backend and display the response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                # Send the POST request to the backend
                response = requests.post(FASTAPI_BACKEND_URL, json=payload)
                
                # Check for any HTTP errors
                response.raise_for_status() 
                
                # Parse the JSON response
                data = response.json()
                answer = data.get("answer", "Sorry, I couldn't get a valid response from the backend.")

                # Display the assistant's response
                st.markdown(answer)
                
                # 4. Add the assistant's response to session state
                st.session_state.messages.append({"role": "assistant", "content": answer})

            except requests.exceptions.ConnectionError:
                st.error(f"**Connection Error:** Could not connect to the backend at `{FASTAPI_BACKEND_URL}`. \n\n**Is your FastAPI server running?**")
            except requests.exceptions.RequestException as e:
                st.error(f"An error occurred while communicating with the backend: {e}")