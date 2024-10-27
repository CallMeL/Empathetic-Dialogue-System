import torch.nn as nn

class StrokeModel(nn.Module):
    def __init__(self):
        pass
    
    def tokenize_data(self):
        pass

    def compute_metrics(self):
        return {"accuracy": -1, "f1": -1}

    def train(self):
        pass

    def predict(self, texts):
        pass

    def save_model(self, model_path):
        self.model.save_pretrained(model_path)

