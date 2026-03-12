import pandas as pd
import numpy as np
import os

# Set up paths
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
outputs_path = os.path.join(project_root, "outputs")

# Load cleaned historical data
historical = pd.read_csv(os.path.join(outputs_path, "cleaned_timeseries.csv"))

# Load forecast data
forecast = pd.read_csv(os.path.join(outputs_path, "future_forecast.csv"))

print("Data loaded successfully.")

#STEP 2: CALCULATE DEMAND VARIABILITY
# Standard deviation of historical demand
demand_std = historical['Weekly_Sales'].std()

print("Demand Standard Deviation:", demand_std)


#STEP 3: DEFINE BUSINESS ASSUMPTIONS
lead_time_weeks = 2      # Supplier takes 2 weeks
service_level = 1.65     # 95% service level (Z-score)


#STEP 4: CALCULATE SAFETY STOCK
# SafetyStock=Z×σ×√(LeadTime)
safety_stock = service_level * demand_std * np.sqrt(lead_time_weeks)

print("Safety Stock Level:", safety_stock)

#STEP 5: CALCULATE REORDER POINT
#ReorderPoint=(AverageDemand×LeadTime)+SafetyStock
average_demand = historical['Weekly_Sales'].mean()

reorder_point = (average_demand * lead_time_weeks) + safety_stock

print("Reorder Point:", reorder_point)


#STEP 6: APPLY TO FORECAST
forecast['Reorder_Point'] = reorder_point
forecast['Safety_Stock'] = safety_stock

forecast['Reorder_Recommendation'] = np.where(
    forecast['Forecasted_Sales'] > reorder_point,
    "Monitor",
    "Reorder"
)

forecast.to_csv(os.path.join(outputs_path, "inventory_recommendation.csv"), index=False)

print("Inventory optimization completed.")
