import streamlit as st
import speech_recognition as sr
from cerebras_call import initial_user_notification, talk_to_user

def display_chatbot():
    st.title("Condition Detected")

    if "messages" not in st.session_state:
        st.session_state["messages"] = [{"role": "assistant", "content": initial_user_notification()}]
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    
    if st.button("Start Speaking"):
        user_input = speech_to_text()
        if user_input:
            process_input(user_input)
        else:
            st.error("No input received")

    if prompt := st.chat_input("Type your message"):
        process_input(prompt)

def process_input(user_input):
    
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)
    
    response = talk_to_user(user_input)
    st.session_state.messages.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.markdown(response)

    st.rerun()

def speech_to_text():
    recognizer = sr.Recognizer()
    
    with st.spinner('Listening...'):
        try:
            with sr.Microphone() as source:
                audio = recognizer.listen(source, timeout=5)
                text = recognizer.recognize_google(audio)
                return text
        except Exception as e:
            st.error(f"An error occurred: {e}")
    
    return None

def main():
    display_chatbot()

main()
