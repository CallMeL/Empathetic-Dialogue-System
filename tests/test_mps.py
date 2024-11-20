# *********************************************************************************************************************
#  Test Mac's GPU
# *********************************************************************************************************************

import torch

def test_mps():
    print(f"PyTorch Version: {torch.__version__}")
    print(f"MPS Available: {torch.backends.mps.is_available()}")
    print(f"MPS Built: {torch.backends.mps.is_built()}")

    if torch.backends.mps.is_available():
        device = torch.device("mps")
        x = torch.ones(5, device=device)
        y = torch.ones(5, device=device)
        z = x + y
        print(f"Tensor z on MPS: {z}")
    else:
        print("MPS is not available. Running on CPU.")
        device = torch.device("cpu")
        x = torch.ones(5, device=device)
        y = torch.ones(5, device=device)
        z = x + y
        print(f"Tensor z on CPU: {z}")

if __name__ == "__main__":
    test_mps()