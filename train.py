# train.py
import time
import torch
import torch.nn as nn
from torch.nn import functional as F
from label_data import prepare_and_label_data # Import from your file

# ==========================================
# 1. CONFIGURATION & CONFIG TRACKING
# ==========================================
filename = "alphabet_data.txt"
batch_size = 32      
block_size = 8        
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'

MAX_CYCLES = 2000      
MAX_TIME_SECONDS = 30 
EVAL_INTERVAL = 200    

# Use your modular function to set up data structures
data_tensor, vocab_size, stoi, itos = prepare_and_label_data(filename, block_size, batch_size)
print(f"Data Loaded Successfully! Total characters in file: {len(data_tensor)}")

# Split into Train and Validation chunks
n = int(0.9 * len(data_tensor))
train_data = data_tensor[:n]
val_data = data_tensor[n:]

def get_batch(split):
    dataset = train_data if split == 'train' else val_data
    ix = torch.randint(len(dataset) - block_size, (batch_size,))
    x = torch.stack([dataset[i:i+block_size] for i in ix])
    y = torch.stack([dataset[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)

# ==========================================
# 2. MODEL DEFINITION
# ==========================================
class AlphabetLanguageModel(nn.Module):
    def __init__(self, vocab_size): # Pass vocab_size dynamically
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, 64)
        self.position_embedding_table = nn.Embedding(block_size, 64)
        self.lm_head = nn.Linear(64, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = tok_emb + pos_emb
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

# ==========================================
# 3. TRAINING LOOP
# ==========================================
if __name__ == "__main__":
    model = AlphabetLanguageModel(vocab_size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    decode = lambda l: ''.join([itos[i] for i in l])

    print(f"\n--- Starting Training (Will auto-kill after {MAX_TIME_SECONDS} seconds) ---")
    start_time = time.time()

    for cycle in range(MAX_CYCLES):
        current_time = time.time()
        elapsed_time = current_time - start_time
        
        if elapsed_time >= MAX_TIME_SECONDS:
            print(f"\n🛑 HARD STOP TRIGGERED: Reached time boundary limit of {MAX_TIME_SECONDS}s.")
            break

        xb, yb = get_batch('train')
        logits, loss = model(xb, yb)
        
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if cycle % EVAL_INTERVAL == 0 or cycle == MAX_CYCLES - 1:
            avg_time_per_cycle = elapsed_time / (cycle + 1)
            remaining_cycles = MAX_CYCLES - (cycle + 1)
            estimated_remaining_time = remaining_cycles * avg_time_per_cycle
            
            preds = torch.argmax(logits, dim=-1)
            correct = (preds == yb.view(-1)).sum().item()
            accuracy = (correct / yb.numel()) * 100

            print(f"\n🔄 CYCLE: {cycle}/{MAX_CYCLES} | Time Elapsed: {elapsed_time:.1f}s")
            print(f"⏱️ Est. Time Remaining: {estimated_remaining_time:.1f}s")
            print(f"📉 Loss: {loss.item():.4f} | Live Batch Accuracy: {accuracy:.1f}%")
            
            test_context = torch.tensor([[stoi['a']]], dtype=torch.long, device=device)
            generated_output = model.generate(test_context, max_new_tokens=15).tolist()
            # Add [0] here to flatten the batch dimension from [[...]] to [...]
            print(f"🔮 Output prediction sample: '{decode(generated_output[0]).strip()}'")


    # Save weights & token metrics for usage after training completes
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'vocab_size': vocab_size,
        'stoi': stoi,
        'itos': itos
    }
    torch.save(checkpoint, "alphabet_model.pt")
    print("\n💾 Progress saved successfully to 'alphabet_model.pt'")
