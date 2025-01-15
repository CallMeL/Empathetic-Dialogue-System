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
        'description': " We trained this model on on Facebook Emotion Dialogues dataset with additional GPT data, using a batch size of 256.",
        'logo': '🧃'
    },
    "single_conversation_withGPTdata_withoutemotion": {
        'url': 'https://huggingface.co/HannahLin271/NanoGPT/resolve/main/singleConversation_withGPTdata_withoutemotion.pt',
        'description': " We trained this model on Facebook Emotion Dialogues dataset with GPT data, excluding emotion tag, using a default batch size of 64.",
        'logo': '🧉'
    },
    "single_conversation_withcontext": {
        'url': 'https://huggingface.co/HannahLin271/NanoGPT/resolve/main/singleConversation_withcontext.pt',
        'description': "Trained on Facebook Emotion Dialogues dataset with context included for improved conversational understanding, using a default batch size of 64.",
        'logo': '🍹'
    },
    "single_conversation_withemotion": {
        'url': 'https://huggingface.co/HannahLin271/NanoGPT/resolve/main/singleConversation_withemotion.pt',
        'description': "Trained on Facebook Emotion Dialogues dataset, retaining emotion annotations for each conversation, using a default batch size of 64.",
        'logo': '🍺'
    },
    "single_conversation": {
        'url': 'https://huggingface.co/HannahLin271/NanoGPT/resolve/main/singleConversation.pt',
        'description': "Trained on Facebook Emotion Dialogues dataset, excluding emotion annotations for simpler conversations, using a default batch size of 64.",
        'logo': '🍷'
    },
    "whole_conversation": {
        'url': 'https://huggingface.co/HannahLin271/NanoGPT/resolve/main/wholeConversation.pt',
        'description': "Trained on entire conversations from the Facebook Emotion Dialogues dataset, excluding tags other than <bot> and <human>,, using a default batch size of 64",
        'logo': '🍵'
    }
}
model_list = { }
model_choices = list(model_info.keys())
# init model for default selection
selected_model_name = "single_conversation_withGPTdata_bs256"
url = model_info[selected_model_name]['url']
model_list[selected_model_name] = init_model_from(url, selected_model_name)

# gpt-2 encodings
print("loading GPT-2 encodings...")
enc = tiktoken.get_encoding("gpt2")
encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
decode = lambda l: enc.decode(l)


def predict(input_hints, input: str,  history: list = None) -> tuple:
    if history is None:
        history = []  # Initialize history if not provided
    # Generate a response using the respond function
    print(f"selected_model_name: {selected_model_name}")
    response_data = respond(
        input=input,
        samples=1,
        model=model_list[selected_model_name],
        encode=encode,
        decode=decode,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
    )
    
    response = response_data[1]
    full_output =  response_data[2]
    print(f"full_output: {full_output}")
    history.append((input, response))  # Append the user input and bot response to history

    return history, history, full_output  # Return updated history twice (for chatbot and state)

def prepare_model(selected_model):
    global selected_model_name
    selected_model_name = selected_model
    url = model_info[selected_model]['url']
    if selected_model not in model_list:
        model_list[selected_model] = init_model_from(url, selected_model)
    logo = model_info[selected_model]['logo']
    description = model_info[selected_model]['description']
    return f"## {logo}Model Information\n<br>Model_name: {selected_model}\n<br>Description: {description}"

def update_chat_with_model_selection(model, chat_history):
    # Add a message about the selected model
    if chat_history is None:
        chat_history = []
    chat_history.append(
        (None, 
         f'<span style="background-color: #FFD700; padding: 4px; border-radius: 4px;">Now you are chatting with <strong>{model}</strong></span>')
    )
    return chat_history


default_model_info = f"## 🍭Model Information\n<br>Model_name: single_conversation_withGPTdata_bs256\n<br>Description: We trained this model on Facebook Emotion Dialogues dataset with additional GPT data, using a batch size of 256."
app = gr.Blocks()
full_output = " "
with app:
    gr.Markdown("# 🫂 Empathetic Dialogue System \n🤗For the chatbot, we trained a small language model from scratch in our local machine. You can find the detailed information about our project in the Github repository [here](https://github.com/CallMeL/Project-ML).")
    # Model Parameters interface
    inp = gr.Dropdown(
            choices=model_choices,
            label="Select a Model",
            info="Choose a pre-trained model to power the chatbot."
        )
    out = gr.Markdown(value=default_model_info)
    inp.change(prepare_model, inp, out)

    # Chatbot interface
    chatbot = gr.Chatbot(
        label="Chatbot Response",
        avatar_images=(
            None,  # User avatar (None for default)
            "https://em-content.zobj.net/source/twitter/376/hugging-face_1f917.png"  # Assistant avatar
        )
    )
    user_input = gr.Textbox(lines=2, placeholder="Enter your message here...", label="User Input")
    state = gr.State([])
    debug_result = gr.Textbox(label="Debug: Full model output",value=full_output)
    input_hints = gr.Markdown("## 📝 Input Hints\n<br>1. Select a model from the dropdown list. \n<br> 2. Type your message in the text box, please try to input a complete sentence.\n<br> 3. Fill the [form](https://forms.office.com/e/PuTy4jrcQD) here to help us evaluate the model")
    chat_interface = gr.Interface(
        fn=predict,
        inputs=[
            input_hints,
            user_input,
            state,  # Maintain conversation state
            
        ],
        outputs=[
            chatbot,
            state,
            debug_result      
        ],
        description="Your AI-based chatbot powered by selected models!"
    )
    
    inp.change(fn=update_chat_with_model_selection, inputs=[inp, state], outputs=[chatbot])
    
    #TODO: add emotion/context here
if __name__ == "__main__":
    app.launch(share=True)