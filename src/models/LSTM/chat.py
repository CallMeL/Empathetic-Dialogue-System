import tiktoken
import torch
from model import LSTMModel

batch_size = 16
block_size = 1024
device = 'cpu'

input_size = 1024
hidden_size = 1024 
num_layers = 12

enc = tiktoken.get_encoding("gpt2")
encode = lambda s: enc.encode(s, allowed_special={""})
decode = lambda l: enc.decode(l)

with open("./input.text","r") as f:   
    test_input_text = f.read()
encoded_input = encode(test_input_text)
padded_encoded_input = encoded_input + [0] * (input_size - len(encoded_input))
test_input_tensor = torch.tensor(padded_encoded_input).to(torch.float32) / 65535
test_input_tensor = test_input_tensor.to(device).unsqueeze(0)

model = LSTMModel(input_size, hidden_size, num_layers).to(device)
model.load_state_dict(torch.load("./out/model.pt"))
model = model.to(device)
model.eval()

with torch.no_grad():
    test_output = model(test_input_tensor)

    test_output = (test_output * 65535).to(torch.int64)

    predicted_output = test_output.argmax(dim=-1).squeeze().cpu().tolist()

    if not isinstance(predicted_output, list):
        predicted_output = [predicted_output]

    predicted_text = decode(predicted_output)

    print(f"Input: {test_input_text}")
    print(f"Output: {predicted_text}")