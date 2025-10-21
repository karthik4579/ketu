from utils import generate_chat_response
import streamlit as st

st.set_page_config(page_title="Ketu | LLM Finetuning Assistant", page_icon="🤖")
st.title("Ketu 🤖")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "awaiting_response" not in st.session_state:
    st.session_state.awaiting_response = False

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if len(st.session_state.messages) == 0:
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        response = generate_chat_response(st.session_state.messages)
        message_placeholder.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

if prompt := st.chat_input("Type your message here..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
        st.session_state.awaiting_response = True
    
if (st.session_state.messages and st.session_state.awaiting_response == True):
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        response = generate_chat_response(st.session_state.messages)
        message_placeholder.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
    st.session_state.awaiting_response = False
    
    