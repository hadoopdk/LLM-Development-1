# interact.py
import os
import torch
import re
from train import AlphabetLanguageModel 

device = 'cuda' if torch.cuda.is_available() else 'cpu'
checkpoint_path = "dictionary_model.pt"

if not os.path.exists(checkpoint_path):
    print(f"❌ Error: Model weights file '{checkpoint_path}' not found! Run 'python train.py' first.")
    exit()

# 1. Load trained conversational data
checkpoint = torch.load(checkpoint_path, map_location=device)
vocab_size = checkpoint['vocab_size']
stoi = checkpoint['stoi']
itos = checkpoint['itos']
decode = lambda l: ' '.join([itos[i] for i in l]).replace(' \n', '\n')

# 2. Build model architecture
model = AlphabetLanguageModel(vocab_size).to(device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print("\n🤖 --- Smart Alphabet AI Ready ---")
print("Ask questions like: 'a for what?' or 'what comes after b?'")
print("Type 'exit' to turn off.")

while True:
    user_prompt = input("\nEnter your question: ").strip().lower()
    if user_prompt == 'exit':
        break
    
    # 1. Auto-append a question mark if the user forgot it
    if not user_prompt.endswith('?') and not user_prompt.endswith('.'):
        user_prompt += ' ?'
        
    # Clean and split the sentence into isolated word tokens
    prompt_words = re.findall(r"\w+|[?.]", user_prompt)
    
    # Check if any entered words are missing from our model's vocabulary
    unknown_words = [w for w in prompt_words if w not in stoi]
    if unknown_words:
        print(f"⚠️ I don't know the word(s): {', '.join(unknown_words)}")
        print("💡 Try using clean phrases like: 'a for what?' or 'what comes before z?'")
        continue
        
    valid_tokens = [stoi[w] for w in prompt_words if w in stoi]
    context_tensor = torch.tensor([valid_tokens], dtype=torch.long, device=device)
    
    # 2. DYNAMIC GENERATION: Generate up to 15 tokens, but stop early if a period '.' is found
    generated_tokens = valid_tokens.copy()
    
    with torch.no_grad():
        for _ in range(15):  # Maximum safety limit
            # Keep context window within the model's block_size limit
            cond_tensor = torch.tensor([generated_tokens[-12:]], dtype=torch.long, device=device)
            logits, _ = model(cond_tensor)
            logits = logits[:, -1, :]
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()
            
            generated_tokens.append(next_token)
            
            # Stop generating immediately if the model finishes its sentence
            if itos[next_token] == '.':
                break
                
    # 3. CLEAN RESPONSE: Unpack full text, then remove the user's question
    full_sentence = decode(generated_tokens)
    clean_question = decode(valid_tokens)
    
    # Extract only the newly generated answer portion
    answer_only = full_sentence.replace(clean_question, "").strip()
    
    print(f"🔮 Response: '{answer_only}'")
