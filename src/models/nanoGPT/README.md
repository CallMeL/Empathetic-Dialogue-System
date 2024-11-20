# NanoGPT
[nanoGPT](https://github.com/karpathy/nanoGPT)
## The architecture of nanoGPT
NanoGPT is a **decoder-only** transformer based language model. Based on the architecture, each subsection contains essential conception to understand how nanoGPT works.
### Encode the input text
* Encoder takes a string and output a list of integers. 
* It is done in the preparation phase (`prepare.py`), will get `.bin` files for training and testing.
* In the code the [tiktoken](https://github.com/openai/tiktoken) from openAI is used.
### Embeddings
1. Token embeddings: represent words in matrix
2. Positional embeddings
   Since the Encoder inside the Transformer simultaneously processes the entire input sequence, the information about the position of the element needs to be encoded inside its embedding. That is why the Positional embedding layer is used, which sums embeddings with a vector of the same dimension: `x = self.transformer.drop(tok_emb + pos_emb`

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
cd ./src/models/nanoGPT/data/shakespeare
python prepare.py
```

3. train the model
`--device` can be set to `cpu` or specially `mps` on MacBooks. 
`max_iters` in the train.py is set to `600000` , it runs forever~ use `^ C` to stop the training at anytime.
```
cd ./src/models/nanoGPT
(add `--init_from=resume \` for continue training)
time python train.py \
  --init_from=resume \
  --n_layer=4 \
  --n_head=4 \
  --n_embd=64 \
  --compile=False \
  --eval_iters=1 \
  --block_size=64 \
  --batch_size=8 \
  --device=mps
```
When the training starts, hit `^ A` so later we can copy all the logs to this [website](https://observablehq.com/@simonw/plot-loss-from-nanogpt), then get the log graph. (we will definitely improve the logging later )

### test the model
### chat with the bot
configure `init_from = 'huggingface' ` or `init_from = 'resume' `
```
cd ./src/models/nanoGPT
 python chat.py
```
###  generate text
```
cd ./src/models/nanoGPT
 python sample.py
```
###

## reference
1. [nanoGPT](https://github.com/karpathy/nanoGPT)
2. [training nanoGPT on MacBook](https://til.simonwillison.net/llms/nanogpt-shakespeare-m2)
3. [Let's build GPT: from scratch, in code, spelled out. (author of nanoGPT)](https://youtu.be/kCc8FmEb1nY?si=XA_iMh2jns5vPHN5)
4. [LLM Visualization](https://bbycroft.net/llm)