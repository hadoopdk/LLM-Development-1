import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import os

# --- MODEL STRUCTURAL CONFIGURATIONS ---
N_EMBED = 128            
N_HEAD = 4               
N_LAYER = 3              
BLOCK_SIZE = 64          
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- 🚨 STRICT INFERENCE FILTERS TO RESOLVE PREFIX ALIGNMENT 🚨 ---
TEMPERATURE = 0.1        # Low randomness forcing absolute logical structure
TOP_K = 2                # Prevent sampling from random tail probabilities
CONFIDENCE_CUTOFF = 0.50 # Filter out noisy field calculations

# Load parameters metadata definitions
if not os.path.exists("model_meta.json") or not os.path.exists("model_weights.pt"):
    print("❌ Error: Missing trained neural state files. Run 'python train_expanded.py' first!")
    exit()

with open("model_meta.json", "r") as f:
    meta = json.load(f)

chars = meta["chars"]
vocab_size = len(chars)
stoi = meta["stoi"]
itos = {int(k): v for k, v in meta["itos"].items()}
encode = lambda s: [stoi[c] for c in s if c in stoi]
decode = lambda l: ''.join([itos[i] for i in l])

# Define matching model architecture class precisely
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(N_EMBED, head_size, bias=False)
        self.query = nn.Linear(N_EMBED, head_size, bias=False)
        self.value = nn.Linear(N_EMBED, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE)))
    def forward(self, x):
        B, T, C = x.shape
        k, q, v = self.key(x), self.query(x), self.value(x)
        wei = q @ k.transpose(-2, -1) * (C**-0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        return wei @ v

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
        self.ln1, self.ln2 = nn.LayerNorm(n_embd), nn.LayerNorm(n_embd)
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
    def forward(self, idx):
        B, T = idx.shape
        x = self.token_embedding_table(idx) + self.position_embedding_table(torch.arange(T, device=DEVICE))
        return self.lm_head(self.ln_f(self.blocks(x)))

# Load model weights layers configurations
model = ScaledGPT().to(DEVICE)
model.load_state_dict(torch.load("model_weights.pt", map_location=DEVICE))
model.eval()

print(f"🚀 Model Weights Successfully Loaded on [{DEVICE.upper()}] (Parameters Optimized)")
print("🤖 --- High-Capacity Transformer Shell Ready ---")
print("Type 'exit' to turn off.\n")

while True:
    user_query = input("Enter your prompt: ").strip().lower()
    if user_query == 'exit': break
    if not user_query: continue

    formatted_prompt = f"user: {user_query} assistant: "
    encoded_input = encode(formatted_prompt)
    if not encoded_input:
        print("🔮 Response: '⚠️ Input characters not found within training vocabulary.'\n")
        continue

    context_tensor = torch.tensor([encoded_input], dtype=torch.long, device=DEVICE)
    
    response_chars = []
    cumulative_confidence = 0.0
    steps = 0

    with torch.no_grad():
        for _ in range(100):  
            context_cond = context_tensor[:, -BLOCK_SIZE:]
            logits = model(context_cond)
            logits = logits[:, -1, :] / TEMPERATURE
            
            v, ix = torch.topk(logits, min(TOP_K, logits.size(-1)))
            logits_filtered = torch.full_like(logits, float('-inf'))
            logits_filtered.scatter_(-1, ix, v)
            
            probs = F.softmax(logits_filtered, dim=-1)
            
            # 🧠 DIAGNOSTIC LOG FOR THE FIRST GENERATED STEP [1]
            if steps == 0:
                top_probs, top_indices = torch.topk(probs, min(3, probs.size(-1)))
                print(f"\n🧠 [Attention Layer Log] Top Probabilities for First Output Token:")
                for p, tidx in zip(top_probs[0], top_indices[0]):
                    print(f"   Character: '{itos[tidx.item()]}' -> Probability Matrix Confidence: {p.item():.4f}")
                print("🔮 Generating response stream...", flush=True)

            max_prob = torch.max(probs).item()
            cumulative_confidence += max_prob
            steps += 1
            
            idx_next = torch.multinomial(probs, num_samples=1)
            context_tensor = torch.cat((context_tensor, idx_next), dim=1)
            
            next_char = itos[idx_next.item()]
            if next_char == '\n': 
                break
            response_chars.append(next_char)

    avg_confidence = cumulative_confidence / max(steps, 1)
    generated_text = "".join(response_chars).strip()

    print("🔮 Response: '", end="")
    if avg_confidence < CONFIDENCE_CUTOFF or len(generated_text) < 2:
        print("🤖 I am sorry, but my network matrix confidence is too low to resolve this prompt safely.")
    else:
        print(generated_text, end="")
    print("'\n")
