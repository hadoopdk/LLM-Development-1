# interact.py
import os
import torch
from train import AlphabetLanguageModel 

device = 'cuda' if torch.cuda.is_available() else 'cpu'
checkpoint_path = "alphabet_model.pt"

if not os.path.exists(checkpoint_path):
    print(f"❌ Error: Model weights file '{checkpoint_path}' not found! Run 'python train.py' first.")
    exit()

# 1. Load parameters and dictionary mappings
checkpoint = torch.load(checkpoint_path, map_location=device)
vocab_size = checkpoint['vocab_size']
stoi = checkpoint['stoi']
itos = checkpoint['itos']
decode = lambda l: ''.join([itos[i] for i in l])

# 2. Build model architecture & plug in trained weights
model = AlphabetLanguageModel(vocab_size).to(device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval() # Secure inference mode

print("\n🤖 --- Language Model Ready ---")
print("Provide letters (e.g., 'abc' or 'rst') and choose your output length.")
print("Type 'exit' to turn off.")

while True:
    user_prompt = input("\nEnter prompt sequence: ").strip().lower()
    if user_prompt == 'exit':
        break
        
    valid_tokens = [stoi[c] for c in user_prompt if c in stoi]
    if not valid_tokens:
        print("⚠️ Characters not recognized by model tokenizer registry.")
        continue
    
    # NEW CUSTOMIZATION PARAMETER: Ask how many characters to add
    length_input = input("How many new characters do you want? (e.g., 3, 5, 10): ").strip()
    if length_input.lower() == 'exit':
        break
        
    try:
        # Convert user text into an integer number
        custom_length = int(length_input)
        if custom_length <= 0:
            print("⚠️ Please enter a number greater than 0.")
            continue
    except ValueError:
        print("⚠️ Invalid number entered. Using default length of 5 instead.")
        custom_length = 5
        
    context_tensor = torch.tensor([valid_tokens], dtype=torch.long, device=device)
    
    with torch.no_grad():
        # Using custom_length directly in the generator setup
        output_indices = model.generate(context_tensor, max_new_tokens=custom_length).tolist()
        
    print(f"🔮 Predicted Text Output: '{decode(output_indices[0]).strip()}'")
