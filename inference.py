import numpy as np
import tensorflow as tf
from train import model, tokenizer, vocab_size, window_size # Importing from your first file

def generate_text(seed_text, max_words=10):
    words = seed_text.lower().split()
    
    # Input validation to ensure the window matches model requirements
    if len(words) < window_size:
        return "Error: Please provide a sentence with at least 5 words."

    generated = words[:]
    
    for i in range(max_words):
        # Extract the last 5 words (sliding window)
        current_window = [generated[-window_size:]]
        seq = tokenizer.texts_to_sequences(current_window)
        
        # safety check for out-of-vocabulary words
        if not seq[0]: break 
        
        # input ready for prediction
        input_data = np.array([tf.keras.utils.to_categorical([idx-1 for idx in seq[0]], num_classes=vocab_size)])
        
        # predict and update
        prediction = model.predict(input_data, verbose=0)
        predicted_idx = np.argmax(prediction) + 1
        predicted_word = tokenizer.index_word[predicted_idx]
        
        generated.append(predicted_word)

    # attached warning for sequence length
    result = " ".join(generated)
    if max_words >= 10:
        result += "\n\n[Note: To generate longer sequences, please change the 'max_words' parameter in the code.]"
    
    return result

# usage
print(generate_text("The man had killed the"))
