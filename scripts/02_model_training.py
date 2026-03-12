import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# Set up paths
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
outputs_path = os.path.join(project_root, "outputs")

# Load cleaned data
df = pd.read_csv(os.path.join(outputs_path, "cleaned_timeseries.csv"))

# Features and target
X = df[['Year', 'Month', 'Week', 'Lag_1']]
y = df['Weekly_Sales']

# Train-test split (time-based split)
split_index = int(len(df) * 0.8)
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Save performance
with open(os.path.join(outputs_path, "model_performance.txt"), "w") as f:
    f.write(f"Mean Absolute Error: {mae}\n")
    f.write(f"R2 Score: {r2}\n")

print("Model training completed.")
print("MAE:", mae)
print("R2:", r2)
