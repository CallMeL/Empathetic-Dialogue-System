import torch
import torch.nn as nn
import inspect


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, batch_first=True):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                            batch_first=batch_first)
        
    def forward(self, x):
        x, _ = self.lstm(x)
        return x
        
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()
        self.lstm1 = LSTM(input_size, hidden_size, num_layers)
        self.linear = nn.Linear(hidden_size, hidden_size)
        self.lstm2 = LSTM(hidden_size, hidden_size, num_layers)
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x):
        x = self.dropout(x)
        x = self.lstm1(x)
        x = self.linear(x)
        x = self.lstm2(x)
        return x
    