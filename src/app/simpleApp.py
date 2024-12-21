import gradio as gr
from utils import init_model_from, respond
import tiktoken
# Configuration
max_new_tokens = 100
temperature = 0.5
top_k = 10

# Model information and links

model_info = {
    "single_conversation_withGPTdata_bs256": {
        'url': 'https://huggingface.co/HannahLin271/NanoGPT/resolve/main/singleConversation_withGPTdata_bs256.pt',
        'description': "Trained on Facebook Emotion Dialogues dataset with additional GPT data, using a batch size of 256.",
        'logo': '🧃'
    },
    "single_conversation_withGPTdata_withoutemotion": {
        'url': 'https://huggingface.co/HannahLin271/NanoGPT/resolve/main/singleConversation_withGPTdata_withoutemotion.pt',
        'description': "Trained on Facebook Emotion Dialogues dataset with GPT data, excluding emotion tag.",
        'logo': '🧉'
    },
    "single_conversation_withcontext": {
        'url': 'https://huggingface.co/HannahLin271/NanoGPT/resolve/main/singleConversation_withcontext.pt',
        'description': "Trained on Facebook Emotion Dialogues dataset with context included for improved conversational understanding.",
        'logo': '🍹'
    },
    "single_conversation_withemotion": {
        'url': 'https://huggingface.co/HannahLin271/NanoGPT/resolve/main/singleConversation_withemotion.pt',
        'description': "Trained on Facebook Emotion Dialogues dataset, retaining emotion annotations for each conversation.",
        'logo': '🍺'
    },
    "single_conversation_withoutemotion": {
        'url': 'https://huggingface.co/HannahLin271/NanoGPT/resolve/main/singleConversation_withoutemotion.pt',
        'description': "Trained on Facebook Emotion Dialogues dataset, excluding emotion annotations for simpler conversations.",
        'logo': '🍷'
    },
    "whole_conversation_withoutemotion": {
        'url': 'https://huggingface.co/HannahLin271/NanoGPT/resolve/main/wholeConversation_withoutemotion.pt',
        'description': "Trained on entire conversations from the Facebook Emotion Dialogues dataset, excluding tags other than <bot> and <human>.",
        'logo': '🍵'
    }
}
model_choices = list(model_info.keys())

# Load model
model = init_model_from("HannahLin271/nanoGPT_single_conversation", "single_conversation_model")
# gpt-2 encodings
print("loading GPT-2 encodings...")
enc = tiktoken.get_encoding("gpt2")
encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
decode = lambda l: enc.decode(l)


def predict(model_selection, input: str, history: list = None) -> tuple:
    print(f"Selected model: {model_selection}")
    if history is None:
        history = []  # Initialize history if not provided

    # Generate a response using the respond function
    response_data = respond(
        input=input,
        samples=1,
        model=model,
        encode=encode,
        decode=decode,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
    )
    
    response = response_data[1]  # Extract bot's response
    history.append((input, response))  # Append the user input and bot response to history

    return history, history  # Return updated history twice (for chatbot and state)

def prepare_model(model_selection):
    return f"Selected model: {model_selection}"

app = gr.Blocks()

with app:
    gr.Markdown("# 🤖 Chatbot for ML Project\n### 🤗🫂 Chat with your ML-based chatbot!")
    # Model Parameters interface
    parameter_interface = gr.Interface(
        fn = prepare_model,
        inputs = [
            gr.Dropdown(
            choices=model_choices,
            value=model_choices[0],  # Default selection
            label="Select a Model",
            info="Choose a pre-trained model to power the chatbot."
        ),
        ],
        outputs=gr.Markdown(label="Model Information"),
    )
    # Chatbot interface
    chat_interface = gr.Interface(
        fn=predict,
        inputs=[
            gr.Textbox(lines=2, placeholder="Enter your message here...", label="User Input"),
            gr.State(),  # Maintain conversation state
        ],
        outputs=[
            gr.Chatbot(label="Chatbot Response"),  # Display responses in chat format
            gr.State()  # Return the updated state
        ],
        description="Your AI-based chatbot powered by selected models!"
    )


if __name__ == "__main__":
    app.launch(share=True)