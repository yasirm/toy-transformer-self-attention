# Toy Transformer Attention Model
A simple yet inside-out implementation of a Self-Attention mechanism for sequence prediction using TensorFlow. The model is trained on really "toy" text snippet - provide it more data to improve it!

## Instructions
1. **Clone the repo:**
2. **Install dependencies:** only TF and numpy
3. **Training:** `python train.py`
4. **Inference:** `python inference.py`

## Key elements
- **Custom Attention Layer:** Built from scratch using Keras Subclassing.
- **Sliding Window:** Predicts the next word based on the previous 5 tokens (which here are same as words).
- **Learnable Positional Encoding:**  Ofc. you may want to change that if you wish.
