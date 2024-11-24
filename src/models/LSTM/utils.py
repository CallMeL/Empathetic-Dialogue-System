import torch
import os
import numpy as np
import math
import json

# saving training process
def save_training_progress(progress, out_dir):
    max_iters = progress["iteration"][-1]
    progress_file = os.path.join(out_dir, f"training_progress_{max_iters}.json")
    # Convert tensors to serializable formats
    for key in progress:
        if isinstance(progress[key], list):
            progress[key] = [float(x) if isinstance(x, torch.Tensor) else x for x in progress[key]]
    with open(progress_file, "w") as f:
        json.dump(progress, f, indent=4)
    print(f"\nProgress saved to {progress_file}")

# learning rate decay scheduler (cosine with warmup)
def get_lr(it,warmup_iters,lr_decay_iters,learning_rate,min_lr):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)

def load_checkpoint(model, optimizer, file_path="model_checkpoint.pth"):
    checkpoint = torch.load(file_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    print(f"Checkpoint loaded from {file_path}")
    return model, optimizer, epoch, loss, 

def save_checkpoint(model, optimizer, epoch, loss, file_path="model_checkpoint.pth"):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'loss': loss
    }
    torch.save(checkpoint, file_path)
    print(f"Checkpoint saved to {file_path}")

def get_batch(split, batch_size=16, block_size=1024, device='cpu',data_dir='../../../data'):
    # We recreate np.memmap every batch to avoid a memory leak, as per
    # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
    if split == 'train':
        data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    else:
        data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
    ix = torch.randint(len(data) - block_size, (batch_size,)) # random offset of the training set
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    loss_mask = torch.ones_like(x, dtype=torch.float32)
    if device == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y, loss_mask = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True), loss_mask.pin_memory().to(device, non_blocking=True)
    else:
        x, y, loss_mask = x.to(device), y.to(device), loss_mask.to(device)
    return x, y