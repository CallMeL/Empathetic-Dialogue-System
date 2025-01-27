# NanoGPT
Code mainly taken from: [nanoGPT](https://github.com/karpathy/nanoGPT)
## A quick cheat sheet to run the code
(make sure the environment is setup, i.e `PYTHONPATH`)
```
cd data
python prepare.py  # Enter the dataset path, default is emotion/with_gpt_data/with_gpt_data.txt

cd ../src/nanoGPT

time python train.py \
  --data_dir=data/emotion/with_gpt_data/ \
  --device=cpu \

# enter control+c to stop the training process, a pt file will be saved under `out` file
python chat.py /Project-ML/src/nanoGPT/out/ckpt.pt #copy the absolute path of the out file

```
## The architecture of nanoGPT
NanoGPT is a **decoder-only** transformer based language model. Based on the architecture, each subsection contains essential conception to understand how nanoGPT works.
### Encode the input text
* Encoder takes a string and output a list of integers. 
* It is done in the preparation phase [`prepare.py`](https://github.com/CallMeL/Project-ML/blob/master/data/prepare.py), will get `.bin` files for training and testing.
* In the code the [tiktoken](https://github.com/openai/tiktoken) from openAI is used.
### Embeddings
1. Token embeddings: represent words in matrix
2. Positional embeddings
+ Since the Encoder inside the Transformer simultaneously processes the entire input sequence, the information about the position of the element needs to be encoded inside its embedding. That is why the Positional embedding layer is used, which sums embeddings with a vector of the same dimension: `x = self.transformer.drop(tok_emb + pos_emb`
+ *. Absolute Positional Embeddings*
    + Assign a unique vector to each position in the input sequence to encode positional information into the model.
    + NanoGPT's default embeddings
2. Positional embeddings (PE)
Since the Encoder inside the Transformer simultaneously processes the entire 
   input sequence, the information about the position of the element needs to be encoded inside its embedding. That is why the Positional embedding layer is used, which sums embeddings with a vector of the same dimension: `x = self.transformer.drop(tok_emb + pos_emb)`

## How can we improve PE?
We can consider these three Positional Embeddings which are mentioned from the talk 04.12
1. ROPE (Rotary Positional Embeddings)
    + Encode positional information by applying rotational transformations to input embeddings
2. Relative Positional Embeddings
    + Instead of encoding the absolute position, focus on the relative distances between tokens in a sequence.


| Feature                     | Absolute Positional Embeddings         | Relative Positional Embeddings         | ROPE                                 |
|-----------------------------|----------------------------------------|----------------------------------------|--------------------------------------|
| **Position Representation** | Unique absolute position               | Relative distances between positions   | Relative information via rotation    |
| **Scalability**             | Limited in some fixed implementations  | Generalizable to varying lengths       | Well-suited for long sequences       |
| **Computational Efficiency**| Simple                                 | Higher complexity                      | Highly efficient                     |
| **Use Cases**               | Standard Transformers                 | NLP tasks with context sensitivity     | Large models like GPTs               |


### Transformer block
Each transformer block contains the following part: When we have the output of one transformer block, we pass it to the next (transformer) block.
#### Layer norm
* This is an operation that normalizes the values in each column of the matrix separately. 
* It helps improve the stability of the model during training.

#### Self-Attention
 It's the phase where the columns in our input embedding matrix "talk" to each other
  * Q: what am I looking for
  * K: what am I containing
  * W: dot product of Q*K (the attention score)

`self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)     `

In the code, flash is used to accelerate the compute process
```
self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
y = torch.nn.functional.scaled_dot_product_attention
```
#### MLP
After the self-attention and a layer normalization, is the MLP (multi-layer perceptron), it's a simple neural network with two layers.
There are 3 steps inside MLP:
1. A linear transformation with a bias added, to a vector of length `4 * config.n_embd`.
2. A GELU activation function (element-wise), introduce some non-linearity into the model.
3. A linear transformation with a bias added, back to a vector of length `config.n_embd`


## other part of the training
### the loss
`loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)` in `model.py`

### the training part
`optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)`

## Run the code
### train the model
always first active the environment with `source venv/bin/activate` and install the requirements.txt. 
if there are some dependency import errors, try `pip install transformers datasets tiktoken tqdm wandb numpy`
1. download pytorch nightly for the first time
```
pip install \
  --pre torch torchvision torchaudio \
  --extra-index-url https://download.pytorch.org/whl/nightly/cpu
```

2. prepare the training data 
```
cd data
python prepare.py
% need to input the txt path like `emotion/robot_human_tag/59k_eachconv_eot.
txt` 
```

3. train the model
`--device` can be set to `cpu` or specially `mps` on MacBooks. 
`max_iters` in the train.py is set to `30100` , it runs forever~ use `^ C` to stop the training at anytime.
```
cd ./src/nanoGPT

# Training for the fist time
time python train.py \
  --data_dir=data/emotion/with_gpt_data/ \
  
# other configuration:
  --n_layer=4 \
  --n_head=4 \
  --n_embd=64 \
  --compile=False \
  --eval_iters=1 \
  --block_size=64 \ # default 64
  --batch_size=8 \
  --device=mps \
  --init_from=resume \ # for continue training
  --pos_embd=rope \ # change the position embeding 
```
+ When the training starts, hit `^ A` so later we can copy all the logs to this [website](https://observablehq.com/@simonw/plot-loss-from-nanogpt), then get the log graph. (we will definitely improve the logging later )
+ pos_embd options: default, rope, relative


### chat with the bot locally
configure `init_from `based on where the trained model is saved
```
cd ./src/models/nanoGPT
python chat.py block_size=64/withoutemotion/wholeConversation
python chat.py block_size=64/withoutemotion/singleConversation
python chat.py block_size=64/withemotion
python chat.py block_size=64/withoutemotion/singleConversation_withGPTdata
python chat.py block_size=64/withcontext
python chat.py block_size=256/singleConversation_withGPTdata
```

###  Evaluate nanoGPT
check src/evaluation/evaluation.ipynb for details.

### Updating Model Checkpoints for Relative Positional Embeddings

When introducing **Relative Positional Embeddings** to an existing model, the structure of the model changes, potentially causing compatibility issues when loading old checkpoint files. Specifically, older checkpoints will not contain the new parameter,  `transformer.relative.relative_embeddings.weight`, leading to errors during the loading process.

To address this issue, the following function was created to update checkpoints by adding the missing parameters in ```ckpt_update.ipynb```

### Explanation

1. **Purpose**  
   The function ensures that old checkpoint files can be loaded into the updated model without errors by checking for and adding the new parameter `transformer.relative.relative_embeddings.weight` if it is missing.

2. **How It Works**  
   - **Checking the State Dictionary:**  
     The function inspects the `state_dict` of the checkpoint for the key `transformer.relative.relative_embeddings.weight`.
   - **Adding Missing Parameter:**  
     If the key is not present, the function initializes the missing parameter as a zero tensor with the same shape as the corresponding weight in the model.
   - **Updating the Checkpoint:**  
     The modified `state_dict` is then reassigned to the checkpoint, ensuring compatibility with the updated model.

3. **Example Output**  
   When a missing parameter is detected and added, the function prints a message to indicate the addition:
   ```
   Adding transformer.relative.relative_embeddings.weight to checkpoint.
   ```

4. **File Naming Convention**  
   - **Old Checkpoints (`ckpt_original.pt`):**  
     These files represent the checkpoint files saved before introducing relative positional embeddings. They do not contain the `transformer.relative.relative_embeddings.weight` parameter.
   - **Updated Checkpoints (`ckpt.pt`):**  
     After running the `update_checkpoint` function, the updated checkpoint files include the required parameter and can be loaded into the new model without errors.

5. **Integration**  
   This function should be applied to all old checkpoint files before loading them into the updated model. This guarantees that the model's structure matches the expected state and avoids runtime errors.

## reference
1. [nanoGPT](https://github.com/karpathy/nanoGPT)
2. [training nanoGPT on MacBook](https://til.simonwillison.net/llms/nanogpt-shakespeare-m2)
3. [Let's build GPT: from scratch, in code, spelled out. (author of nanoGPT)](https://youtu.be/kCc8FmEb1nY?si=XA_iMh2jns5vPHN5)
4. [LLM Visualization](https://bbycroft.net/llm)
