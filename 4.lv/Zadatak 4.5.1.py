import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import sklearn.linear_model as lm
from sklearn.metrics import mean_absolute_error


data = pd.read_csv('data_C02_emission.csv')

# numeričke veličine sa nazivima stupca te podjela 
# na skup učenje i skup za testiranje u omjeru 80%-20%

veličine = [
    "Engine Size (L)",
    "Cylinders",
    "Fuel Consumption City (L/100km)",
    "Fuel Consumption Hwy (L/100km)",
    "Fuel Consumption Comb (L/100km)",
    "Fuel Consumption Comb (mpg)"
]

target = "CO2 Emissions (g/km)"

X = data[veličine]
y = data[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

# dijagram raspršenja prikazati ovisnost CO2 plinova
# o jednoj numeričkoj veličini
plt.figure()
plt.scatter(X_train["Engine Size (L)"], y_train, color = "blue", label = "Train")
plt.scatter(X_test["Engine Size (L)"], y_test, color = "red", label = "Test")
plt.xlabel("Engine Size (L)")
plt.ylabel("CO2 Emissions (g/km)")
plt.title("Ovisnost CO2 plinova o veličini motora")
plt.legend()
plt.show()

# standardizacija ulaznih veličina
scalar = MinMaxScaler()
X_train_scalar = scalar.fit_transform(X_train)
X_test_scalar = scalar.fit_transform(X_test)

# histogram jedne ulazne veličine prije i nakon skaliranja
plt.figure(figsize=(8,6))

# prije
plt.subplot(1, 2, 1)
plt.hist(X_train["Engine Size (L)"], bins = 20)
plt.title("Prikaz Engine Size (L) prije skaliranja")

# nakon
plt.subplot(1, 2, 2)
plt.hist(X_train_scalar[:, 0], bins = 20)
plt.title("Prikaz Engine Size (L) nakon skaliranja")

plt.tight_layout()
plt.show()

# linearni regresijski model
model = lm.LinearRegression()
model.fit(X_train_scalar, y_train)

# Ispis koeficijenata i presjeka modela
print("Dobiveni parametri linearnog modela:")
print(f"θ0 (intercept): {model.intercept_:.4f}")

# Ispis koeficijenata za svaku varijablu
for i, (feature, coef) in enumerate(zip(X_train.columns, model.coef_), start=1):
    print(f"θ{i} * {feature}: {coef:.4f}")

# Generisanje matematičkog izraza modela
equation = f"ŷ = {model.intercept_:.4f} + "
equation += " + ".join([f"{coef:.4f} * {feature}" for coef, feature in zip(model.coef_, X_train.columns)])

print("\nDobiveni model linearnom regresijom (odgovara izrazu 4.6):")
print(equation)

# pomoću dijagrama raspršenja odnos između stvarnih vrijednosti izlazne 
# veličine i procjene dobivene modelom
y_pred = model.predict(X_test_scalar)

plt.figure()
plt.scatter(y_test, y_pred)
plt.title("Odnos između stvarnih i procjenjenih vrijednosti emisije")
plt.show()

# vrijednost regresijske metrike na skupu za testiranje
mea = mean_absolute_error(y_test, y_pred)

mse = np.mean((y_test - y_pred) ** 2)

total = np.sum((y_test - np.mean(y_test)) ** 2)
res = np.sum((y_test - y_pred) ** 2)
r2 = 1 - (res / total)

print(f"MEA: {mea:.2f}")
print(f"MSE: {mse:.2f}")
print(f"R2: {r2:.2f}")

# vrijednost evaluacijske metrike na testom skupu kada se mijenja broj
# ulaznih veličina
X_train_single = X_train_scalar[:, [0]] 
X_test_single = X_test_scalar[:, [0]]

model_single = lm.LinearRegression()
model_single.fit(X_train_single, y_train)

y_pred_single = model_single.predict(X_test_single)

mae_single = mean_absolute_error(y_test, y_pred_single)
print(f"MAE za model s jednom varijablom: {mae_single:.2f}")

model.fit(X_train_scalar, y_train)
y_pred_all = model.predict(X_test_scalar)

mae_all = mean_absolute_error(y_test, y_pred_all)
print(f"MAE za model s više varijabli: {mae_all:.2f}")