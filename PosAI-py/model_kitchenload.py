import pandas as pd
import numpy as np
import tensorflow as tf

# Simulate historical orders: [hour_of_day, day_of_week] → demand for fries
rows = []
for _ in range(500):
    hour = np.random.randint(0, 24)
    day = np.random.randint(0, 7)
    # Fries demand peaks lunch (12-14) and dinner (18-20)
    demand = np.random.poisson(lam=5)
    if 12 <= hour <= 14: demand += np.random.randint(5, 15)
    if 18 <= hour <= 20: demand += np.random.randint(5, 15)
    rows.append([hour, day, demand])

data = pd.DataFrame(rows, columns=["hour", "day", "fries_demand"])

X = data[["hour", "day"]].values.astype(np.float32)
y = data[["fries_demand"]].values.astype(np.float32)

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(2,)),
    tf.keras.layers.Dense(8, activation="relu"),
    tf.keras.layers.Dense(1)
])

model.compile(optimizer="adam", loss="mse")
model.fit(X, y, epochs=100, verbose=0)

# Convert to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open("output_kitchenload.tflite", "wb") as f:
    f.write(tflite_model)

print("✅ output_kitchenload.tflite saved")
