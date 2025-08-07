import pandas as pd
import numpy as np
import tensorflow as tf

# --- Mock POS order dataset ---
# Input: burger, pizza, coffee
# Output: fries, coke, muffin, cookie

data = pd.DataFrame([
    [1,0,0, 1,0,0,0],  # Burger → Fries
    [0,1,0, 0,1,0,0],  # Pizza → Coke
    [0,0,1, 0,0,1,0],  # Coffee → Muffin
    [1,0,1, 0,0,0,1],  # Burger+Coffee → Cookie
    [0,1,1, 0,1,1,0],  # Pizza+Coffee → Coke+Muffin
    [1,1,0, 1,1,0,0],  # Burger+Pizza → Fries+Coke
    [0,0,1, 0,0,0,1],  # Coffee → Cookie
    [1,1,1, 1,1,1,0],  # Burger+Pizza+Coffee → Fries+Coke+Muffin
    [0,1,0, 0,1,0,1],  # Pizza → Coke+Cookie
    [1,0,0, 1,0,0,1],  # Burger → Fries+Cookie
], columns=["burger","pizza","coffee","fries","coke","muffin","cookie"])
# data = pd.read_csv("pos_training_data.csv")

# Features (X) and labels (y)
X = data[["burger", "pizza", "coffee"]].values.astype(np.float32)
y = data[["fries", "coke", "muffin", "cookie"]].values.astype(np.float32)

# --- Model definition ---
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(3,), dtype=tf.float32),
    tf.keras.layers.Dense(8, activation="relu"),
    tf.keras.layers.Dense(4, activation="sigmoid")  # 4 products to recommend
])

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.fit(X, y, epochs=200, verbose=1)

# --- Test prediction ---
test_input = np.array([[1,0,0]], dtype=np.float32)  # Burger only
pred = model.predict(test_input)
print("Prediction for Burger only:", pred)

# --- Save model as .tflite ---
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open("output_predection.tflite", "wb") as f:
    f.write(tflite_model)

print("✅ TFLite model saved as output_predection.tflite")
