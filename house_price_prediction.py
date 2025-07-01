# Enhanced Linear Regression Model for House Price Prediction

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# Load Dataset (Make sure train.csv is in the same folder)
data = pd.read_csv("train.csv")

# Select enhanced feature set
features = ['GrLivArea', 'BedroomAbvGr', 'FullBath', 'GarageArea', 'YearBuilt', 'TotalBsmtSF']
target = 'SalePrice'

# Drop rows with missing values in selected features
data = data.dropna(subset=features + [target])

# Extract features and target
X = data[features]
y = data[target]

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train Linear Regression Model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on Test Set
y_pred = model.predict(X_test)

# Evaluation
print("\nModel Evaluation:")
print("R^2 Score:", r2_score(y_test, y_pred))
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))

# Plot Actual vs Predicted
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=y_pred)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted House Prices")
plt.show()

# Show Coefficients
coeff_df = pd.DataFrame(model.coef_, features, columns=['Coefficient'])
print("\nFeature Coefficients:")
print(coeff_df)

# Export Predictions to CSV
pred_df = pd.DataFrame({
    'ActualPrice': y_test,
    'PredictedPrice': y_pred
})
pred_df.to_csv('predictions.csv', index=False)
print("\nPredictions saved to 'predictions.csv'")

