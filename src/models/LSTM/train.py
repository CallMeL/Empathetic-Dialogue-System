import os
import time

import torch
import torch.nn as nn
from torch.nn import functional as F

from model import LSTMModel 
import tiktoken
from utils import get_batch

data_dir = os.path.join('../../../data')

batch_size = 16
block_size = 1024
device = 'cpu'

input_size = 1024
hidden_size = 1024 
num_layers = 12
print_iter = 10
save_checkpoint = 10

model = LSTMModel(input_size, hidden_size, num_layers).to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())
X, Y = get_batch('train', batch_size, block_size, device, data_dir)


for i in range(20):
    
    # uint160->float32
    X = X.to(torch.float32) / 65535  
    Y = Y.to(torch.float32) / 65535
    
    optimizer.zero_grad()
    output = model(X)
    output = output.to(torch.float32) / 65535  
    Y = Y.to(torch.float32) / 65535
    
    loss = loss_fn(output, Y)
    
    loss.backward()
    optimizer.step()

    if i % print_iter == 0:
        print(f"Step {i}, Loss {loss.item()}")
    if i % save_checkpoint == 0 and i!=0:
        torch.save(model.state_dict(), "./out/model.pt")
        print(f"Model saved at step {i}")
