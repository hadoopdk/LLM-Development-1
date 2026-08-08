import random

alphabet = 'abcdefghijklmnopqrstuvwxyz'

def generate_bulk_corpus():
    print("⏳ Generating ~1.2 million tokens of alphabet reasoning metrics...")
    lines = []
    
    # PHASE 1: DIRECT LOOKUPS (A-Z) with structural diversity
    objects = {
        'a': 'apple', 'b': 'ball', 'c': 'cat', 'd': 'dog', 'e': 'elephant',
        'f': 'fish', 'g': 'gorilla', 'h': 'hat', 'i': 'ice', 'j': 'juice',
        'k': 'kite', 'l': 'lion', 'm': 'monkey', 'n': 'nest', 'o': 'owl',
        'p': 'penguin', 'q': 'queen', 'r': 'rabbit', 's': 'snake', 't': 'tiger',
        'u': 'umbrella', 'v': 'violin', 'w': 'water', 'x': 'xylophone',
        'y': 'yak', 'z': 'zebra'
    }
    
    templates_definition = [
        "user: {letter} for what? assistant: {letter} is for {obj}",
        "user: tell me what {letter} stands for assistant: {letter} is for {obj}",
        "user: what is {letter} for assistant: {letter} is for {obj}",
        "user: {letter} fo what assistant: {letter} is for {obj}",
        "user: {letter} for assistant: {letter} is for {obj}"
    ]
    
    for _ in range(8000):
        letter = random.choice(alphabet)
        temp = random.choice(templates_definition)
        lines.append(temp.format(letter=letter, obj=objects[letter]))

    # PHASE 2: SINGLE SEQUENCING MECHANICS (After & Before)
    templates_sequence = [
        "user: what after {char1} assistant: {char2} comes after {char1}",
        "user: what comes after {char1} assistant: {char2} comes after {char1}",
        "user: what letter comes after {char1} assistant: {char2} comes after {char1}",
        "user: what following {char1} assistant: {char2} comes after {char1}",
        "user: what next to {char1} assistant: {char2} comes after {char1}",
        "user: what abter {char1} assistant: {char2} comes after {char1}",
        "user: what before {char2} assistant: {char1} comes before {char2}",
        "user: what comes before {char2} assistant: {char1} comes before {char2}",
        "user: what precedes {char2} assistant: {char1} comes before {char2}"
    ]
    
    for _ in range(15000):
        idx = random.randint(0, len(alphabet) - 2)
        char1 = alphabet[idx]
        char2 = alphabet[idx + 1]
        temp = random.choice(templates_sequence)
        lines.append(temp.format(char1=char1, char2=char2))

    # PHASE 3: MULTI-CONDITION REASONING (Chain-of-Thought)
    templates_logic = [
        "user: what after {char1} and before {char3} assistant: thought: after {char1} is {char2}. before {char3} is {char2}. match found. result: {char2} comes after {char1} and before {char3}",
        "user: what comes after {char1} and before {char3} assistant: thought: after {char1} is {char2}. before {char3} is {char2}. match found. result: {char2} comes after {char1} and before {char3}",
        "user: what abter {char1} and before {char3} assistant: thought: after {char1} is {char2}. before {char3} is {char2}. match found. result: {char2} comes after {char1} and before {char3}"
    ]

    for _ in range(15000):
        idx = random.randint(0, len(alphabet) - 3)
        char1 = alphabet[idx]
        char2 = alphabet[idx + 1]
        char3 = alphabet[idx + 2]
        temp = random.choice(templates_logic)
        lines.append(temp.format(char1=char1, char2=char2, char3=char3))

    random.shuffle(lines)
    output_text = "\n".join(lines)
    token_count = len(output_text.split()) + len(output_text) // 4
    
    with open("corpus_large.txt", "w", encoding="utf-8") as f:
        f.write(output_text)
        
    print(f"🚀 Success! Created 'corpus_large.txt' containing {len(lines)} data rows.")
    print(f"📊 Matrix Density: Approximately {token_count:,} structural tokens generated.")

if __name__ == "__main__":
    generate_bulk_corpus()
