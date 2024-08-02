
import torch
import torch.nn as nn
from torch.nn import functional as F

"""### Hyperparameters"""

'''Hyperparameters for smaller model'''

B = 32 # B: how many independent sequences will we process in parallel?
T = 8  # T: what is the maximum context length for predictions?
C = 32 # C: numer of different features analysed (also D = dims)
H = 4  # H: number of attention heads
L = 4  # L: Number of layers
learning_rate = 1e-3

'''Final Hyperparameters'''

# B = 64 # B: how many independent sequences will we process in parallel?
# T = 256  # T: what is the maximum context length for predictions?
# H = 6
# C = 64*H
# L = 6
# learning_rate = 1e-4

# Common Hyperparameters
max_iters = 5000
eval_interval = 500
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
dropout = 0.2
torch.manual_seed(1337)

with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# here are all the unique characters that occur in this text
text = text.replace('\n', ' ').replace('\n\n', ' ')
chars = sorted(list(set(text.split(" "))))
vocab_size = len(chars)
# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s.split(" ")] # encoder: take a string, output a list of integers
decode = lambda l: ' '.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

chars_str = ''.join(chars)
print(f'vocab_size: {vocab_size}')
print(f'vocabulary: {chars_str}')

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long, device=device)
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - T, (B,))
    x = torch.stack([data[i:i+T] for i in ix])
    y = torch.stack([data[i+1:i+T+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

"""### Head, MHSA"""

class Head(nn.Module):
    """ One head of self attention"""

    def __init__(self, Ci, Co):
        super().__init__()
        self.key   = nn.Linear(Ci, Co, bias=False)
        self.query = nn.Linear(Ci, Co, bias=False)
        self.value = nn.Linear(Ci, Co, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(T, T)))

    def forward(self, x):
        B, T, Ci  = x.shape
        '''
        B  - batch               # of independant vectors processed
        T  - time/block/context  # of tokens in a context
        Ci - channals/dims input # of features in input
        '''

        k = self.key(x)   # (B,T,Co)
        q = self.query(x) # (B,T,Co)

        # compute attention scores / affinities
        wei = q @ k.transpose(-2,-1)                                 # (B,T,Co) @ (B,Co,T) -> (B,T,T)
        wei /= C**0.5                                                # (B,T,T) scaling, bring variance to 1, to prevent softmax clipping
        wei  = wei.masked_fill(self.tril[:T,:T]==0, float('-inf'))   # (B,T,T) Replace upper triangular of wei with -inf
        wei  = F.softmax(wei, dim=-1)                                # (B,T,T) -inf -> 0, rest normalized to 1

        v = self.value(x)  # (B,T,Co)
        out = wei @ v      # (B,T,T) @ (B,T,Co) = (B,T,Co)

        return out


class MultiHeadAttention(nn.Module):

    def __init__(self, Ci, H, head_size):
        super().__init__()
        # 4 heads of 8-dimensional self-attention, for n_embed=32, like a group convolution
        self.heads = nn.ModuleList([Head(Ci=Ci, Co=head_size) for _ in range(H)])

    def forward(self, x):
        x = torch.cat([h(x) for h in self.heads], dim=-1)
        return x

"""### Transformer Block"""

class Block(nn.Module):
    ''' Transformer block: communication followed by computation '''

    def __init__(self, C, H): # C: embedding dimension, H: number of heads
        super().__init__()
        self.ln1 = nn.LayerNorm(C)   # Layernorm along channels (batch & time are batch dims): y = beta + gamma * [x-E(x)]/sqrt(V(x) + ep)
        self.sa = MultiHeadAttention(Ci=C, H=H, head_size=C//H)
        self.ln2 = nn.LayerNorm(C)
        self.ffwd = nn.Sequential(         # Feedforward network, so the tokens can "think about" what they found in attention.
            nn.Linear(C, C*4),
            nn.GELU(),
            nn.Linear(C*4, C),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        # Residual connections around MSA & FF, to help training
        # Note: input without layernorm is added to output

        x_skip = x

        x = self.ln1(x)
        x = self.sa(x)   # (B,T,C), Multi head self attention
        x = x + x_skip

        x = self.ln2(x)
        x = self.ffwd(x) # (B,T,C), Per token level. B,T act as batch dimensions
        x = x + x_skip

        return x

"""### Model"""

class BigramLanguageModel(nn.Module):

    def __init__(self, B,T,C,H,L):
        super().__init__()
        self.B, self.T, self.C, self.H, self.L = B,T,C,H,L
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, C) # for every possible token, weights for next token
        self.position_embedding_table = nn.Embedding(T, C)

        self.blocks  = nn.Sequential(*[Block(C, H) for _ in range(L)])
        self.ln_final = nn.LayerNorm(C)
        self.lm_head = nn.Linear(C, vocab_size)

    def forward(self, idx, targets=None):

        tok_emb = self.token_embedding_table(idx)                                    # (B,T,C=n_embed)
        pos_emb = self.position_embedding_table(torch.arange(self.T, device=device)) # (T,C): [0,1,2..T-1]

        x = tok_emb + pos_emb     # (B,T,C)
        x = self.blocks(x)
        x = self.ln_final(x)      # Layernorm applied before last
        logits = self.lm_head(x)  # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):                        # idx is (B, T) array of indices in the current context
            idx_cond = idx[:, -self.T:]                        # crop the last block_size tokens for input
            logits, loss = self(idx_cond)                      # get the predictions
            logits = logits[:, -1, :]                          # (B,T,C) -> (B, C)
            probs = F.softmax(logits, dim=-1)                  # (B, C)
            idx_next = torch.multinomial(probs, num_samples=1) # sample from the distribution acc to prob (B, 1)
            idx = torch.cat((idx, idx_next), dim=1)            # New idx is concat (B, T+1)
        return idx

model = BigramLanguageModel(B,T,C,H,L)
m = model.to(device)

"""### Training"""

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):
    if iter % eval_interval == 0:   # every once in a while evaluate the loss on train and val sets
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    xb, yb = get_batch('train')     # sample a batch of data

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

"""### Inference"""

context = torch.ones((1, T), dtype=torch.long, device=device)  # start with '\n\n\n\n' as seed
out_ints = m.generate(context, max_new_tokens=2000)[0].tolist() # output list of ints
print(decode(out_ints))

