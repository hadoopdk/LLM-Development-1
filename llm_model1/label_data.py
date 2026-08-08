# label_data.py
import os
import torch

def prepare_and_label_data(file_path="alphabet_data.txt", block_size=8, batch_size=4):
    """
    Reads a plain text file, tokenizes characters, and labels the data
    by creating matching input (X) and target (Y) pairs.
    """
    # 1. Read the plain text file
    if not os.path.exists(file_path):
        print(f"❌ Error: File '{file_path}' not found. Creating a sample file for you...")
        with open(file_path, "w", encoding="utf-8") as f:
            f.write("abcdefghijklmnopqrstuvwxyz\n" * 100) # Increased count for training pool
    
    with open(file_path, "r", encoding="utf-8") as f:
        raw_text = f.read()

    # 2. Build the vocabulary mapping (Tokenizer)
    unique_chars = sorted(list(set(raw_text)))
    vocab_size = len(unique_chars)
    string_to_int = { ch:i for i,ch in enumerate(unique_chars) }
    int_to_string = { i:ch for i,ch in enumerate(unique_chars) }

    # 3. Numeric Tokenization
    encoded_text = [string_to_int[char] for char in raw_text]
    data_tensor = torch.tensor(encoded_text, dtype=torch.long)

    # Return everything needed by train.py and interact.py
    return data_tensor, vocab_size, string_to_int, int_to_string
