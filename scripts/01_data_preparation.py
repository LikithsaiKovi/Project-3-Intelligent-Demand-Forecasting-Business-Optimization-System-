import pandas as pd
import os

# Set up paths
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path = os.path.join(project_root, "data", "walmart-sales-dataset-of-45stores.csv")
outputs_path = os.path.join(project_root, "outputs")

# Create outputs directory if it doesn't exist
os.makedirs(outputs_path, exist_ok=True)

# Load dataset
df = pd.read_csv(data_path)

# Convert Date column (handle dd-mm-yyyy format)
df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')

# Aggregate total weekly sales
total_sales = df.groupby('Date')['Weekly_Sales'].sum().reset_index()

# Sort by date
total_sales = total_sales.sort_values('Date')

# Create time features
total_sales['Year'] = total_sales['Date'].dt.year
total_sales['Month'] = total_sales['Date'].dt.month
total_sales['Week'] = total_sales['Date'].dt.isocalendar().week.astype(int)

# Create Lag feature
total_sales['Lag_1'] = total_sales['Weekly_Sales'].shift(1)

# Drop missing values
total_sales = total_sales.dropna()

# Save cleaned dataset
total_sales.to_csv(os.path.join(outputs_path, "cleaned_timeseries.csv"), index=False)

print("Data preparation completed successfully.")
