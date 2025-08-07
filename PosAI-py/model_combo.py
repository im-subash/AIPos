import pandas as pd
import numpy as np
import tensorflow as tf

# Menu items
inputs = ["burger", "pizza", "coffee"]
outputs = ["fries", "coke", "muffin", "cookie", "garlic_bread", "salad"]

# Recommendation rules
def recommend(order):
    fries = 1 if order[0] == 1 else 0  # Burger → Fries
    coke = 1 if order[1] == 1 else 0   # Pizza → Coke
    muffin = 1 if order[2] == 1 else 0 # Coffee → Muffin
    cookie = 1 if (order[0] and order[2]) else 0  # Burger+Coffee → Cookie
    garlic_bread = 1 if order[1] == 1 else 0      # Pizza → Garlic Bread
    salad = np.random.choice([0,1], p=[0.8, 0.2]) # Random occasional suggestion
    return [fries, coke, muffin, cookie, garlic_bread, salad]

# Generate dataset
rows = []
for _ in range(500):
    order = [np.random.choice([0,1]) for _ in inputs]
    recommendation = recommend(order)
    rows.append(order + recommendation)

# Create DataFrame
columns = inputs + outputs
data = pd.DataFrame(rows, columns=columns)

# Train model
X = data[inputs].values.astype(np.float32)
y = data[outputs].values.astype(np.float32)

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(len(inputs),)),
    tf.keras.layers.Dense(16, activation="relu"),
    tf.keras.layers.Dense(len(outputs), activation="sigmoid")
])

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.fit(X, y, epochs=150, verbose=0)

# Convert to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open("output_combo.tflite", "wb") as f:
    f.write(tflite_model)

print("✅ output_combo.tflite saved")
