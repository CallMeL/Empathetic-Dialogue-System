from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import gradio as gr

tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-large")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-large")

def predict(input, history=[]):
    new_user_input_ids = tokenizer.encode(input + tokenizer.eos_token, return_tensors='pt')
    bot_input_ids = torch.cat([torch.LongTensor(history), new_user_input_ids], dim=-1)
    history = model.generate(bot_input_ids, max_length=4000, pad_token_id=tokenizer.eos_token_id).tolist()
    response = tokenizer.decode(history[0]).split("<|endoftext|>")
    response = [(response[i], response[i+1]) for i in range(0, len(response)-1, 2)]
    return response, history

# gr.Interface(fn=predict,
#              inputs=["text", "state"],
#              outputs=["chatbot", "state"]).launch()

app = gr.Interface(
    fn=predict,
    inputs=[
        gr.Textbox(lines=2, placeholder="Enter your message here...", label="User Input"),
        "state"
    ],
    outputs=["chatbot", "state"],
    title="🤖Chatbot for ML project",
    description="🤗🫂",
)

if __name__ == "__main__":
    app.launch(share=True)