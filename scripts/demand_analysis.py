import pandas as pd
import os

# Load dataset
data_path = os.path.join(os.path.dirname(__file__), "..", "data", "walmart-sales-dataset-of-45stores.csv")
df = pd.read_csv(data_path)

# Convert Date column
df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')

print(df.head())
print(df.shape)

# Aggregate total weekly sales across all stores
total_sales = df.groupby('Date')['Weekly_Sales'].sum().reset_index()

print(total_sales.head())

#STEP 3: Sort by Date
total_sales = total_sales.sort_values('Date')


#STEP 4: Basic Visualization
import matplotlib.pyplot as plt

plt.figure(figsize=(10,5))
plt.plot(total_sales['Date'], total_sales['Weekly_Sales'])
plt.title("Total Weekly Sales (All Stores Combined)")
plt.xlabel("Date")
plt.ylabel("Total Sales")
plt.show()

#STEP 5: Create Time-Based Features
total_sales['Year'] = total_sales['Date'].dt.year
total_sales['Month'] = total_sales['Date'].dt.month
total_sales['Week'] = total_sales['Date'].dt.isocalendar().week

#STEP 6: Create Lag Feature
total_sales['Lag_1'] = total_sales['Weekly_Sales'].shift(1)
total_sales = total_sales.dropna()
