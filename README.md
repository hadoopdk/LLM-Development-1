1. Install torch:
   >pip install torch

2. Train Model:
   >python train.py

   Data Loaded Successfully! Total characters in file: 2700

--- Starting Training (Will auto-kill after 30 seconds) ---

🔄 CYCLE: 0/2000 | Time Elapsed: 0.0s
⏱️ Est. Time Remaining: 0.0s
📉 Loss: 3.6268 | Live Batch Accuracy: 3.9%
🔮 Output prediction sample: 'aqhiuvrcaogonsss'

🔄 CYCLE: 200/2000 | Time Elapsed: 0.5s
⏱️ Est. Time Remaining: 4.1s
📉 Loss: 0.0960 | Live Batch Accuracy: 100.0%
🔮 Output prediction sample: 'abcdefghijklmnop'

🔄 CYCLE: 1999/2000 | Time Elapsed: 4.8s
⏱️ Est. Time Remaining: 0.0s
📉 Loss: 0.0014 | Live Batch Accuracy: 100.0%
🔮 Output prediction sample: 'abcdefghijklmnop'

💾 Progress saved successfully to 'alphabet_model.pt'


   
3. Run Inference:
   >python interact.py

   Data Loaded Successfully! Total characters in file: 2700

🤖 --- Language Model Ready ---
Provide letters (e.g., 'abc' or 'rst') and choose your output length.
Type 'exit' to turn off.

Enter prompt sequence: a
How many new characters do you want? (e.g., 3, 5, 10): 1
🔮 Predicted Text Output: 'ab'

Enter prompt sequence: a
How many new characters do you want? (e.g., 3, 5, 10): 3
🔮 Predicted Text Output: 'abcd'

Enter prompt sequence: f
How many new characters do you want? (e.g., 3, 5, 10): 2
🔮 Predicted Text Output: 'fgh'

Enter prompt sequence:


Note: for inference it ask char/string & number of char to predict like(a & 2, it will predict abc)
