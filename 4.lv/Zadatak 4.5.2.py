import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import sklearn.linear_model as lm
from sklearn.metrics import mean_absolute_error, max_error

# Učitavanje podataka
data = pd.read_csv('data_C02_emission.csv')

# Definiranje numeričkih i kategorijskih varijabli
numeric_features = [
    "Engine Size (L)",
    "Cylinders",
    "Fuel Consumption City (L/100km)",
    "Fuel Consumption Hwy (L/100km)",
    "Fuel Consumption Comb (L/100km)",
    "Fuel Consumption Comb (mpg)"
]

categorical_features = ["Fuel Type"]
target = "CO2 Emissions (g/km)"

encoder = OneHotEncoder(drop="first", sparse_output=False)  # drop="first" izbjegava višestruku kolinearost
encoded_categories = encoder.fit_transform(data[categorical_features])

category_names = encoder.get_feature_names_out(categorical_features)

X = np.hstack((data[numeric_features].values, encoded_categories))
X_columns = numeric_features + list(category_names)

X = pd.DataFrame(X, columns=X_columns)

y = data[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = lm.LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = np.mean((y_test - y_pred) ** 2)
max_err = max_error(y_test, y_pred)

index_max_error = np.argmax(abs(y_test - y_pred))
car_with_max_error = data.iloc[X_test.index[index_max_error]]

print(f"MAE: {mae:.2f}")
print(f"MSE: {mse:.2f}")
print(f"Maksimalna pogreška u procjeni CO2 emisije: {max_err:.2f} g/km")
print("\nModel vozila s najvećom pogreškom:")
print(car_with_max_error.to_string())