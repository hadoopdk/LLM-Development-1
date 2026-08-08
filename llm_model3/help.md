Run 
python generate_data.py

python train_cpu.py

python interactpy

===================================

AlphabetGPT: A From-Scratch GPT-2 Architecture in Pure PyTorch

AlphabetGPT is a lightweight, character-level **Decoder-Only Generative Pre-trained Transformer** built entirely from scratch using pure PyTorch. The project architecture is structurally identical to OpenAI's **GPT-2**, implemented efficiently to train on a CPU or GPU using a custom, high-density algorithmic dataset.

The model is designed to showcase how modern Large Language Models (LLMs) solve linguistic variety, handle human typos (e.g., `abter` -> `after`), and execute multi-condition logical operations through **Chain-of-Thought (CoT)** reasoning.

---

## 🚀 Key Architectural Similarities to GPT-2

This repository implements the exact mathematical formulas and layer ordering detailed in the original GPT-2 research paper:

*   **Pre-Layer Normalization (Pre-LN):** Layer normalization blocks are applied *before* the self-attention and feed-forward tracks to optimize gradient flow and training stability.
*   **Causal Self-Attention Matrix:** Utilizes multi-head attention combined with a lower-triangular causal mask matrix (`torch.tril`) to ensure tokens only look backward in time.
*   **GELU Activation Functions:** Uses Gaussian Error Linear Units (`nn.GELU`) instead of traditional ReLU within the Feed-Forward networks to capture higher-dimensional non-linear patterns.
*   **Autoregressive Inference Engine:** Generates response streams character-by-character, dynamically feeding its own predictions back into its context window.

---

## 📊 Model Specifications

| Architectural Parameter | Configuration Value |
| :--- | :--- |
| **Context Window (`BLOCK_SIZE`)** | 64 characters |
| **Embedding Channels (`N_EMBED`)** | 128 dimensions |
| **Attention Heads (`N_HEAD`)** | 4 parallel heads |
| **Transformer Layers (`N_LAYER`)** | 3 blocks |
| **Optimization Mechanics** | AdamW with unbuffered real-time flushing |
| **Total Parameter Count** | ~500,000 trainable weights |

---

## 🛠️ File Structure

*   `generate_data.py`: High-density synthetic data engine that generates ~1.2 million tokens of conversational, lookup, and logical reasoning permutations.
*   `train_expanded.py`: The CPU-optimized core deep learning pipeline featuring real-time training tickers and try/catch backpropagation wrappers.
*   `interact.py`: The production-grade inference engine fitted with rigorous deployment filters (**Temperature scaling**, **Top-K filtering**, and a **Confidence matrix watchdog**).

---

## 🚀 Quick Start Guide

### 1. Installation
Clone the repository and install the standard deep learning framework requirements:
```bash
pip install torch
```

### 2. Generate the Dataset (~1.2M Tokens)
Compile the multi-condition synthetic logic dataset into your folder workspace:
```bash
python generate_data.py
```
This automatically outputs a comprehensive text data file named `corpus_large.txt`.

### 3. Execute Neural Network Training
Run backpropagation loops to train your Transformer's weight parameters:
```bash
python train_expanded.py
```
*Thanks to forced output flushing, your console will show live status tracker prints every second without freezing.*

### 4. Boot the Production Inference Shell
Deploy your trained checkpoint matrices and begin chatting with your model:
```bash
python interact.py
```

---

## 🔮 Sample Queries & Generalization Results

Because the self-attention heads learn abstract token proximity geometry instead of static text strings, the model is resilient to messy human typos and complex constraints:

```text
Enter your prompt: what abter h and before j
🧠 [Attention Layer Log] Top Probabilities for First Output Token:
   Character: 'i' -> Probability Matrix Confidence: 0.9842
🔮 Generating response stream...
Response: 'i comes after h and before j'

Enter your prompt: m for
Response: 'm is for monkey'
```

---

## 🛡️ Production Inference Safety Filters

To bridge the final gap between a raw network and an industry-grade chatbot deployment gateway, `interact.py` enforces three execution guardrails:
1.  **Strict Temperature Scaling (0.1):** flattens choice variance and forces the model to focus strictly on structural language patterns.
2.  **Top-K Constraint (2):** Prevents the model from sampling from lower-ranked noise tails in the token probability distribution.
3.  **Confidence Watchdog (0.50):** If a user types a completely unknown prompt outside the dataset parameters, the model's average softmax field collapses. The watchdog flags this drop immediately and gracefully states: `🤖 I am sorry, but my network matrix confidence is too low to resolve this prompt safely.`