import os
import requests
import tiktoken
import numpy as np

train_ids=[]
val_ids=[]
enc = tiktoken.get_encoding("gpt2")

#TODO: upload our dataset to huggingface and use the link here
# def download_file(url):
#   response = requests.get(url)
#   if response.status_code == 200:
#     with open('dataset.txt', 'wb') as f:
#       f.write(response.content)
#       print("downloaded dataset, tokenizing")
#   else:
#     print('Error downloading file:', response.status_code)

#download_file('https://huggingface.co/VatsaDev/ChatGpt-nano/resolve/main/Dataset.txt')

# MARK: Change the path to the dataset
# TODO: configure the path to the dataset
folder_path = 'data/emotion/with_gpt_data'
input_file_path = os.path.join(folder_path, 'with_gpt_data.txt')
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
train_ids.tofile(os.path.join(folder_path, 'train.bin'))
val_ids.tofile(os.path.join(folder_path, 'val.bin'))