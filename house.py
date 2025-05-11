import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# Load the original DataFrame
original_df = pd.read_csv('Housing.csv')

# Create a numeric-only version
numeric_df = original_df.select_dtypes(include=['number'])


# Filter the numeric DataFrame based on price
filtered_numeric = numeric_df[numeric_df['area'] <= 4000]

# Use the index of the filtered rows to subset the original DataFrame
filtered_original = original_df.loc[filtered_numeric.index]

# Split the data into training and testing sets
X = filtered_original[['area', 'bedrooms', 'bathrooms']]
y = filtered_original['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=3)

#Linear regression model
model = LinearRegression()
model.fit(X_train, y_train)
# Predict prices for the test set
y_pred = model.predict(X_test)

#Add predicted prices to the dataframe
filtered_original['predicted_price'] = model.predict(X)


# Evaluate the model
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'Root Mean Squared Error: {mse**.5}')
print(f'Mean Absolute Error: {mae}')
print(f'R-squared: {r2}')

plt.scatter(y_test, y_pred)
plt.axline((0, 0), (100000, 100000), linewidth=1, color='r')
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Actual vs Predicted Prices')
plt.show()

#Sample 5 poor predictions
test_indices = X_test.index
test = filtered_original.loc[test_indices]
poor_predictions = test[abs(test['price'] - test["predicted_price"])  > 100000].sample(5)
print(poor_predictions[['price', 'predicted_price', 'bedrooms', "bathrooms", "area"]])

# Print coefficients and intercept
for feature, coef in zip(X.columns, model.coef_[:3]):
    print(f"{feature}: {round(coef, 5)}")
print("Intercept:", model.intercept_)

coefficients = pd.DataFrame({"Feature": X.columns, "Coefficient": model.coef_})
coefficients.to_csv("coefficients.csv", index=False)


















