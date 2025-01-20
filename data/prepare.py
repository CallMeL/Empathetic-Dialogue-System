import os
import tiktoken
import numpy as np
from pathlib import Path

train_ids=[]
val_ids=[]
enc = tiktoken.get_encoding("gpt2")

if __name__ == "__main__":
    current_path = Path().absolute()
    print(f"Current absolute path: {current_path}")
    # Change the path to the dataset
    folder_path = input(
        "Enter the dataset name (default: "
        "'emotion/with_gpt_data/with_gpt_data.txt'): ").strip()
    if not folder_path:
        folder_path = 'emotion/with_gpt_data/with_gpt_data.txt'
    directory_path = os.path.dirname(folder_path)
    input_file_path = Path(folder_path)
    print(f"Input file path: {input_file_path}")
    with open(input_file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # Split based on lines instead of characters
    split_idx = int(len(lines) * 0.9)
    train_data = ''.join(lines[:split_idx])
    val_data = ''.join(lines[split_idx:])


# Save the validation data to a file to use it in src/evaluation.ipynb
with open('validation_data.txt', 'w', encoding='utf-8') as val_file:
    val_file.write(val_data)

# encode with tiktoken gpt2 bpe
train_ids = enc.encode_ordinary(train_data)
val_ids = enc.encode_ordinary(val_data)
print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")

# export to bin files
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)
train_ids.tofile(os.path.join(directory_path, 'train.bin'))
val_ids.tofile(os.path.join(directory_path, 'val.bin'))