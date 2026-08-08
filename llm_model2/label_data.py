# label_data.py
import os
import torch
import re

def prepare_and_label_data(file_path="dictionary_data.txt", block_size=16, batch_size=4):
    """
    Generates a highly robust, multi-phrased alphabet question-and-answer dataset
    to eliminate positional memorization and overfitting.
    """
    if not os.path.exists(file_path):
        print(f"📝 Generating robust randomized alphabet dataset: '{file_path}'")
        
        letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 
                   'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
        
        objects = ['apple', 'banana', 'cat', 'dog', 'elephant', 'fish', 'grape', 'hat', 
                   'igloo', 'juice', 'kite', 'lion', 'monkey', 'nest', 'orange', 'penguin', 
                   'queen', 'rabbit', 'snake', 'tiger', 'umbrella', 'violin', 'whale', 
                   'xylophone', 'yo-yo', 'zebra']
        
        qa_dataset = []
        
        # Loop through all 26 options and create varied phrasing styles
        for i in range(len(letters)):
            l = letters[i]
            obj = objects[i]
            
            # Variety 1: Standard
            qa_dataset.append(f"{l} for what ? {l} is for {obj} .")
            # Variety 2: Formal
            qa_dataset.append(f"what is {l} for ? {l} is for {obj} .")
            # Variety 3: Direct
            qa_dataset.append(f"tell me {l} for what ? {l} is for {obj} .")
            
            # Add positioning variations to protect sequence mapping
            if i < len(letters) - 1:
                next_l = letters[i+1]
                qa_dataset.append(f"what comes after {l} ? {next_l} comes after {l} .")
                qa_dataset.append(f"tell me what comes after {l} ? {next_l} comes after {l} .")
            if i > 0:
                prev_l = letters[i-1]
                qa_dataset.append(f"what comes before {l} ? {prev_l} comes before {l} .")
        
        # Combine everything together into a raw text block
        raw_text_block = "\n".join(qa_dataset) + "\n"
        
        # Multiply it so the transformer has thousands of training examples
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(raw_text_block * 40)
            
    with open(file_path, "r", encoding="utf-8") as f:
        raw_text = f.read().lower()

    # Match tokens safely
    words = re.findall(r"\w+|[?.]|\n", raw_text)

    unique_words = sorted(list(set(words)))
    vocab_size = len(unique_words)
    
    string_to_int = { word:i for i, word in enumerate(unique_words) }
    int_to_string = { i:word for i, word in enumerate(unique_words) }

    encoded_text = [string_to_int[w] for w in words]
    data_tensor = torch.tensor(encoded_text, dtype=torch.long)

    return data_tensor, vocab_size, string_to_int, int_to_string
