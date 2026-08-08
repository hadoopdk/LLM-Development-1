# train.py
import time
import torch
import torch.nn as nn
from torch.nn import functional as F
from label_data import prepare_and_label_data

filename = "dictionary_data.txt"
batch_size = 64        # Larger batch for more stable training gradient updates
block_size = 16       
learning_rate = 1e-3  
device = 'cuda' if torch.cuda.is_available() else 'cpu'

MAX_CYCLES = 3500      # Slightly extended cycles for robust pattern absorption
MAX_TIME_SECONDS = 90  
EVAL_INTERVAL = 500    

data_tensor, vocab_size, stoi, itos = prepare_and_label_data(filename, block_size, batch_size)
print(f"🚀 Data Loaded! Vocabulary size: {vocab_size} unique words.")

n = int(0.9 * len(data_tensor))
train_data = data_tensor[:n]
val_data = data_tensor[n:]

def get_batch(split):
    dataset = train_data if split == 'train' else val_data
    ix = torch.randint(len(dataset) - block_size, (batch_size,))
    x = torch.stack([dataset[i:i+block_size] for i in ix])
    y = torch.stack([dataset[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)

class CausalSelfAttention(nn.Module):
    def __init__(self, head_size=128):
        super().__init__()
        self.key = nn.Linear(128, head_size, bias=False)
        self.query = nn.Linear(128, head_size, bias=False)
        self.value = nn.Linear(128, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.proj = nn.Linear(head_size, 128)
        # Added Dropout for regularization
        self.attn_dropout = nn.Dropout(0.1)
        self.resid_dropout = nn.Dropout(0.1)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)   
        q = self.query(x) 
        
        wei = q @ k.transpose(-2, -1) * (k.shape[-1] ** -0.5) 
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) 
        wei = F.softmax(wei, dim=-1)
        wei = self.attn_dropout(wei) # Drop focus points randomly
        
        v = self.value(x) 
        out = wei @ v     
        return self.resid_dropout(self.proj(out))

class AlphabetLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, 128)
        self.position_embedding_table = nn.Embedding(block_size, 128)
        
        self.attention = CausalSelfAttention(head_size=128)
        self.ln1 = nn.LayerNorm(128)
        
        self.ffwd = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.Dropout(0.1) # Added dropout element to hidden processing nodes
        )
        self.ln2 = nn.LayerNorm(128)
        self.lm_head = nn.Linear(128, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx) 
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) 
        
        x = tok_emb + pos_emb 
        x = x + self.attention(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        logits = self.lm_head(x) 

        loss = None
        if targets is not None:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

if __name__ == "__main__":
    model = AlphabetLanguageModel(vocab_size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    decode = lambda l: ' '.join([itos[i] for i in l]).replace(' \n ', '\n')

    print(f"\n--- Starting Robust Regularized Transformer Loop ---")
    start_time = time.time()

    for cycle in range(MAX_CYCLES):
        elapsed_time = time.time() - start_time
        if elapsed_time >= MAX_TIME_SECONDS:
            print(f"\n🛑 HARD STOP TRIGGERED at {MAX_TIME_SECONDS}s.")
            break

        xb, yb = get_batch('train')
        logits, loss = model(xb, yb)
        
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if cycle % EVAL_INTERVAL == 0 or cycle == MAX_CYCLES - 1:
            preds = torch.argmax(logits, dim=-1)
            accuracy = ((preds == yb.view(-1)).sum().item() / yb.numel()) * 100
            print(f"🔄 Cycle: {cycle}/{MAX_CYCLES} | Loss: {loss.item():.4f} | Training Accuracy: {accuracy:.1f}%")

    checkpoint = {
        'model_state_dict': model.state_dict(),
        'vocab_size': vocab_size,
        'stoi': stoi,
        'itos': itos
    }
    torch.save(checkpoint, "dictionary_model.pt")
    print("\n💾 Generalized robust weights saved successfully!")
