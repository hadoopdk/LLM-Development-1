import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import os
import time
import sys

# --- CPU-BALANCED BALANCED HYPERPARAMETERS ---
BATCH_SIZE = 32        
BLOCK_SIZE = 64        
MAX_ITERS = 4000       # Increased from 1500 to let weights resolve specific letter alignment
LEARNING_RATE = 1e-3   
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

N_EMBED = 128          
N_HEAD = 4             
N_LAYER = 3            
DROPOUT = 0.1          

torch.manual_seed(1337)

# --- DATA PIPELINE ---
DATASET_FILE = "corpus_large.txt"
if not os.path.exists(DATASET_FILE):
    print("❌ Error: 'corpus_large.txt' not found. Run generate_data.py first!", flush=True)
    exit()

print(f"📖 Loading dataset from '{DATASET_FILE}'...", flush=True)
with open(DATASET_FILE, "r", encoding="utf-8") as f:
    text = f.read().lower()

chars = sorted(list(set(text)))
vocab_size = len(chars)
print(f"📦 Unique character tokens discovered: {vocab_size}", flush=True)
print(f"📊 Total character count in file: {len(text)}", flush=True)

stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s if c in stoi]

print("⚡ Converting text corpus into numeric tensor space...", flush=True)
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

def get_batch(split):
    data_set = train_data if split == 'train' else val_data
    ix = torch.randint(0, len(data_set) - BLOCK_SIZE, (BATCH_SIZE,))
    x = torch.stack([data_set[i:i+BLOCK_SIZE] for i in ix])
    y = torch.stack([data_set[i+1:i+BLOCK_SIZE+1] for i in ix])
    return x.to(DEVICE), y.to(DEVICE)

@torch.no_grad()
def estimate_loss():
    model.eval()
    out = {}
    for split in ['train', 'val']:
        losses = torch.zeros(5) 
        for k in range(5):
            X, Y = get_batch(split)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# --- ARCHITECTURE MODULES ---
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(N_EMBED, head_size, bias=False)
        self.query = nn.Linear(N_EMBED, head_size, bias=False)
        self.value = nn.Linear(N_EMBED, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE)))
        self.dropout = nn.Dropout(DROPOUT)
    def forward(self, x):
        B, T, C = x.shape
        wei = self.query(x) @ self.key(x).transpose(-2, -1) * (C**-0.5) 
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) 
        return F.softmax(wei, dim=-1) @ self.value(x)

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(N_EMBED, N_EMBED)
    def forward(self, x): return self.proj(torch.cat([h(x) for h in self.heads], dim=-1))

class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_embd, 4 * n_embd), nn.GELU(), nn.Linear(4 * n_embd, n_embd))
    def forward(self, x): return self.net(x)

class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.sa = MultiHeadAttention(n_head, n_embd // n_head)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
    def forward(self, x):
        return x + self.ffwd(self.ln2(x + self.sa(self.ln1(x))))

class ScaledGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, N_EMBED)
        self.position_embedding_table = nn.Embedding(BLOCK_SIZE, N_EMBED)
        self.blocks = nn.Sequential(*[Block(N_EMBED, n_head=N_HEAD) for _ in range(N_LAYER)])
        self.ln_f = nn.LayerNorm(N_EMBED) 
        self.lm_head = nn.Linear(N_EMBED, vocab_size)
    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.ln_f(self.blocks(self.token_embedding_table(idx) + self.position_embedding_table(torch.arange(T, device=DEVICE))))
        logits = self.lm_head(x) 
        loss = F.cross_entropy(logits.view(B*T, logits.shape[-1]), targets.view(B*T)) if targets is not None else None
        return logits, loss

# --- TRAINING TIMELINE DEPLOYMENT ---
print(f"🖥️ Model built on accelerator: [{DEVICE.upper()}]", flush=True)
model = ScaledGPT().to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

print("🏋️‍♂️ Training started. Watching steps live...", flush=True)
start_time = time.time()

try:
    for iter in range(MAX_ITERS):
        xb, yb = get_batch('train')
        logits, loss = model(xb, yb)
        
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        # Real-time ticking feedback line
        if iter % 10 == 0:
            print(f"       ⏱️ Step Tracker: {iter}/{MAX_ITERS} | Active Batch Loss: {loss.item():.4f}", end="\r", flush=True)

        # Clear logging dump blocks
        if iter % 250 == 0 or iter == MAX_ITERS - 1:
            print(f"\n📈 Checkpoint reached at step {iter}...", flush=True)
            eval_losses = estimate_loss()
            elapsed = (time.time() - start_time) / 60
            print(f"   📊 Metrics -> Train Loss: {eval_losses['train']:.4f} | Val Loss: {eval_losses['val']:.4f} | Time Elapsed: {elapsed:.2f} mins", flush=True)

    torch.save(model.state_dict(), "model_weights.pt")
    with open("model_meta.json", "w") as f:
        json.dump({"chars": chars, "stoi": stoi, "itos": itos}, f)
    print("\n🚀 Training successfully completed without system lockups!", flush=True)

except Exception as e:
    print(f"\n❌ CRASH DETECTED IN TRAINING LAYER: {str(e)}", flush=True)
except KeyboardInterrupt:
    print("\n🛑 Training process terminated manually by user command.", flush=True)
