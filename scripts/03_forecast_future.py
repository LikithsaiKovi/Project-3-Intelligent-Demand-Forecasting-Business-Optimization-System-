import pandas as pd
import os
from sklearn.linear_model import LinearRegression

# Set up paths
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
outputs_path = os.path.join(project_root, "outputs")

# Load cleaned data
df = pd.read_csv(os.path.join(outputs_path, "cleaned_timeseries.csv"))

# Features and target
X = df[['Year', 'Month', 'Week', 'Lag_1']]
y = df['Weekly_Sales']

# Train model on full dataset
model = LinearRegression()
model.fit(X, y)

# Prepare future dates
last_date = pd.to_datetime(df['Date'].max())
future_dates = pd.date_range(start=last_date, periods=13, freq='W')[1:]

future_df = pd.DataFrame({'Date': future_dates})
future_df['Year'] = future_df['Date'].dt.year
future_df['Month'] = future_df['Date'].dt.month
future_df['Week'] = future_df['Date'].dt.isocalendar().week.astype(int)

# Use last known sales as Lag_1 start
last_sales = df['Weekly_Sales'].iloc[-1]
future_df['Lag_1'] = last_sales

# Predict
future_df['Forecasted_Sales'] = model.predict(
    future_df[['Year', 'Month', 'Week', 'Lag_1']]
)

# Save forecast
future_df.to_csv(os.path.join(outputs_path, "future_forecast.csv"), index=False)

print("Future forecast generated successfully.")
