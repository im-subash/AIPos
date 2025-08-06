import pandas as pd
import numpy as np

# Menu items
inputs = ["burger", "pizza", "coffee"]
outputs = ["fries", "coke", "muffin", "cookie"]

# Rules for recommendations (more realistic)
def recommend(order):
    fries = 1 if order[0] == 1 else np.random.choice([0,1], p=[0.8, 0.2])  # Burger → Fries (80% chance)
    coke = 1 if order[1] == 1 else np.random.choice([0,1], p=[0.85, 0.15])  # Pizza → Coke (85% chance)
    muffin = 1 if order[2] == 1 else np.random.choice([0,1], p=[0.85, 0.15]) # Coffee → Muffin (85% chance)
    cookie = 1 if (order[0] and order[2]) else np.random.choice([0,1], p=[0.9, 0.1]) # Burger+Coffee → Cookie
    return [fries, coke, muffin, cookie]

# Generate dataset
rows = []
for _ in range(200):
    order = [
        np.random.choice([0,1]),  # Burger
        np.random.choice([0,1]),  # Pizza
        np.random.choice([0,1])   # Coffee
    ]
    recommendation = recommend(order)
    rows.append(order + recommendation)

# Create DataFrame
columns = inputs + outputs
data = pd.DataFrame(rows, columns=columns)

# Save to CSV for inspection (optional)
data.to_csv("pos_training_data.csv", index=False)

print(data.head(10))
print(f"✅ Generated {len(data)} training rows and saved to pos_training_data.csv")
