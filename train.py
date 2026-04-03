import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Layer, Input, Dense, Flatten, Embedding
from tensorflow.keras.models import Model
import numpy as np

# --- Data Preparation (change this block if you have lots & lots of data; at present it's just Mr. Wilde's couplet ---
paragraph = "The man had killed the thing he loved and so he had to die"
words = paragraph.lower().split()
window_size = 5
#------

train_data = []
for i in range(len(words) - window_size):
    window = words[i:i+window_size]
    label = words[i+window_size]
    train_data.append((window, label))

tokenizer = Tokenizer()
tokenizer.fit_on_texts([paragraph])
vocab_size = len(tokenizer.word_index)

# Converting text to sequences
sequences = tokenizer.texts_to_sequences([w for w, l in train_data])
labels = [tokenizer.word_index[l] for _, l in train_data]

# tensor conversion
train_X = np.array([to_categorical([i-1 for i in seq], num_classes=vocab_size) for seq in sequences])
train_y = to_categorical([i-1 for i in labels], num_classes=vocab_size)

# --- Custom Attention Layer ---
class AttentionLayer(Layer):
    def __init__(self, dim):
        super(AttentionLayer, self).__init__()
        self.dim = tf.cast(dim, tf.float32)

    def call(self, inputs):
        Q, K, V = inputs
        # Standard Scaled Dot-Product Attention formula
        scores = tf.matmul(Q, K, transpose_b=True) / tf.sqrt(self.dim)
        weights = tf.nn.softmax(scores, axis=-1)
        return tf.matmul(weights, V)

# --- actual model ---
input_shape = (window_size, vocab_size)
input_layer = Input(shape=input_shape)

# a simple Learnable Position Bias (could ur fav. method too) 
# Instead of manual increments, we add a trainable parameter to let the model learn word order
pos_indices = tf.range(start=0, limit=window_size, delta=1)
pos_encoding = Embedding(input_dim=window_size, output_dim=vocab_size)(pos_indices)
x = input_layer + pos_encoding 

# Projection layers for Q, K, V
Q_bar = Dense(32, activation='relu')(x) # Increased units slightly for better capacity
K_bar = Dense(32, activation='relu')(x)
V_bar = Dense(32, activation='relu')(x)

attention_out = AttentionLayer(dim=32)([Q_bar, K_bar, V_bar])


x = Flatten()(attention_out)
x = Dense(64, activation='relu')(x)
x = tf.keras.layers.Dropout(0.2)(x) 

output_layer = Dense(vocab_size, activation='softmax')(x)

# training ....
model = Model(inputs=input_layer, outputs=output_layer)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(train_X, train_y, epochs=50, verbose=1)

model.save('toy_transformer.h5') # Saves architecture + weights
import pickle
with open('tokenizer.pkl', 'wb') as f:
    pickle.dump(tokenizer, f) # Saves word-to-index mapping
