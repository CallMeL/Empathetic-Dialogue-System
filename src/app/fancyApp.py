import os
import json
import streamlit as st
from huggingface_hub import InferenceApi, login, InferenceClient


# Streamlit app configuration
st.set_page_config(page_title="Chatbot for ML project")
st.title(" Empathetic Dialogue Chatbot 🤖")
st.subheader("", divider='rainbow')
# Get the Hugging Face token from environment variables
hf_token = os.getenv("HF_TOKEN")
if hf_token is None:
    raise ValueError("Hugging Face token not found. Please set the HF_TOKEN environment variable.")

# Authenticate with Hugging Face
login(hf_token)

# Model information and links
model_links = {
    "Zephyr-7B": "HuggingFaceH4/zephyr-7b-beta"
}
model_info = {
    "Zephyr-7B": {
        'description': """Zephyr 7B is a Huggingface model, fine-tuned for helpful and instructive interactions.""",
        # 'logo': 'https://huggingface.co/HuggingFaceH4/zephyr-7b-gemma-v0.1/resolve/main/thumbnail.png'
        'logo': 'https://huggingface.co/HuggingFaceH4/zephyr-7b-alpha/resolve/main/thumbnail.png'
    }
}

# Sidebar for model selection
#st.sidebar.image(model_info['Zephyr-7B']['logo'])
selected_model = st.sidebar.selectbox("Select Model", model_links.keys())
st.sidebar.write(f"You're now chatting with **{selected_model}**")
st.sidebar.markdown(model_info[selected_model]['description'])

# Inference API Initialization
client = InferenceClient('HuggingFaceH4/zephyr-7b-beta')

# Sidebar settings
max_tokens = st.sidebar.slider("Max new tokens", 1, 2048, 512)
temperature = 0.7#st.sidebar.slider("Temperature", 0.1, 4.0, 0.7)
top_p = 0.95#st.sidebar.slider("Top-p (nucleus sampling)", 0.1, 1.0, 0.95)

# Reset conversation button
def reset_conversation():
    st.session_state.messages = []
    st.session_state.model = selected_model

st.sidebar.button('Reset Chat', on_click=reset_conversation)

# Initialize conversation and chat history
if 'messages' not in st.session_state:
    st.session_state.messages = [
        {"role": "system", "content": "I am a empathetic intelligent assistant providing support to you"}
    ]

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

def respond(message, history, max_tokens, temperature, top_p):
    # Prepare the list of messages for the chat completion
    messages = [{"role": "system", "content": st.session_state.messages[0]["content"]}]

    for val in history:
        if val["role"] == "user":
            messages.append({"role": "user", "content": val["content"]})
        elif val["role"] == "assistant":
            messages.append({"role": "assistant", "content": val["content"]})

    messages.append({"role": "user", "content": message})

    # Generate response
    response = ""
    response_container = st.empty()  # Placeholder to update the response text dynamically

    for message in client.chat_completion(
        messages,
        max_tokens=max_tokens,
        stream=True,
        temperature=temperature,
        top_p=top_p,
    ):
        token = message.choices[0].delta.content
        response += token
        response_container.text(response)  # Stream the response

    return response


# User input
if user_input := st.chat_input("Ask a health question..."):
    # Display user message
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Generate and display assistant response
    response = respond(user_input, st.session_state.messages, max_tokens, temperature, top_p)
    with st.chat_message("assistant"):
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})
    st.rerun()