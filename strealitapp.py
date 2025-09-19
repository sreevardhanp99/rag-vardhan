import streamlit as st
import requests
import json
from typing import List, Dict
from datetime import datetime
 
# Configure Streamlit page
st.set_page_config(
    page_title="RAG Document Q&A",
    layout="wide"
)
 
# API Configuration
API_BASE_URL = "http://localhost:8010"
 
# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
 
def upload_documents(files) -> Dict:
    """Upload documents to the FastAPI backend"""
    try:
        files_data = []
        for file in files:
            files_data.append(("files", (file.name, file.getvalue(), file.type)))
       
        response = requests.post(f"{API_BASE_URL}/upload", files=files_data)
       
        if response.status_code == 200:
            return {"success": True, "data": response.json()}
        else:
            return {"success": False, "error": response.json().get("detail", "Upload failed")}
   
    except requests.exceptions.ConnectionError:
        return {"success": False, "error": "Cannot connect to server. Make sure FastAPI is running on port 8010."}
    except Exception as e:
        return {"success": False, "error": str(e)}
 
def ask_question_with_memory(question: str) -> str:
    """Ask question with chat history for memory"""
    try:
        # Get last 6 messages for memory
        memory_messages = st.session_state.chat_history[-6:] if len(st.session_state.chat_history) > 6 else st.session_state.chat_history
       
        # Format memory for the prompt
        memory_context = ""
        if memory_messages:
            memory_context = "\n\nPrevious conversation:\n"
            for i, msg in enumerate(memory_messages):
                if msg["role"] == "user":
                    memory_context += f"Q: {msg['content']}\n"
                else:
                    memory_context += f"A: {msg['content']}\n"
       
        enhanced_question = f"{question}{memory_context}"
       
        response = requests.post(
            f"{API_BASE_URL}/ask",
            json={"question": enhanced_question},
            headers={"Content-Type": "application/json"}
        )
       
        if response.status_code == 200:
            return response.text.strip('"')
        else:
            return f"Error: {response.json().get('detail', 'Unknown error')}"
   
    except requests.exceptions.ConnectionError:
        return "Error: Cannot connect to server."
    except Exception as e:
        return f"Error: {str(e)}"
 
def add_message_to_history(role: str, content: str):
    """Add message to chat history"""
    st.session_state.chat_history.append({
        "role": role,
        "content": content
    })
 
# Main UI
st.title("RAG Document Q&A")
 
# File upload section
st.subheader("Upload Documents")
uploaded_files = st.file_uploader(
    "Choose PDF or DOCX files",
    type=["pdf", "docx"],
    accept_multiple_files=True
)
 
if uploaded_files:
    if st.button("Upload & Process"):
        with st.spinner("Processing documents..."):
            result = upload_documents(uploaded_files)
           
            if result["success"]:
                st.success("Documents uploaded successfully!")
                data = result["data"]
            else:
                st.error(f"Upload failed: {result['error']}")
 
# Chat section
st.subheader("Chat")
 
# Display chat history
if st.session_state.chat_history:
    for message in st.session_state.chat_history:
        if message["role"] == "user":
            with st.chat_message("user"):
                st.write(message["content"])
        else:
            with st.chat_message("assistant"):
                st.write(message["content"])
 
# Question input
question = st.text_input("Ask a question about your documents:")
 
if st.button("Ask") and question.strip():
    # Add user question to history
    add_message_to_history("user", question)
   
    # Show user message immediately
    with st.chat_message("user"):
        st.write(question)
   
    # Get answer
    with st.spinner("Getting answer..."):
        answer = ask_question_with_memory(question)
   
    # Add assistant response to history
    add_message_to_history("assistant", answer)
   
    # Show assistant response
    with st.chat_message("assistant"):
        st.write(answer)
   
    # Rerun to clear input
    st.rerun()
 
# Clear chat button
if st.session_state.chat_history:
    if st.button("Clear Chat"):
        st.session_state.chat_history = []
        st.rerun()