import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score


X, y = make_classification(n_samples=200, n_features=2, n_redundant=0, n_informative=2,
                            random_state=213, n_clusters_per_class=1, class_sep=1)

# train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)

# (a)
plt.figure()
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='coolwarm', label="Train data")
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap='coolwarm', label="Test data")
plt.legend()
plt.title("Podaci za učenje i testiranje")
plt.xlabel("x1")
plt.ylabel("x2")
plt.show()

# (b)
model = LogisticRegression()
model.fit(X_train, y_train)

print(f"θ0 (intercept): {model.intercept_[0]}")
print(f"θ1, θ2 (koeficijenti): {model.coef_[0]}")

# (c)
theta_0 = model.intercept_[0]  # presjek
theta_1, theta_2 = model.coef_[0]  # koeficijenti za x1 i x2

x1_vals = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)
x2_vals = -(theta_0 + theta_1 * x1_vals) / theta_2

plt.figure()
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='coolwarm', label="Train Data")
plt.plot(x1_vals, x2_vals, 'k--', label="Decision Boundary")
plt.xlabel("x1")
plt.ylabel("x2")
plt.legend()
plt.title("Granica odluke modela logističke regresije")
plt.show()

# (d)
y_pred = model.predict(X_test)

# matrica zabune
conf_matrix = confusion_matrix(y_test, y_pred)

# točnost
accuracy = accuracy_score(y_test, y_pred)

# preciznost
precision = precision_score(y_test, y_pred)

# odziv
recall = recall_score(y_test, y_pred)

print("Matrica zabune:")
print(conf_matrix)

print(f"\nTočnost: {accuracy:.4f}")
print(f"Preciznost: {precision:.4f}")
print(f"Odziv: {recall:.4f}")

# (e)
correct = (y_pred == y_test)
incorrect = (y_pred != y_test)

plt.figure()
plt.scatter(X_test[correct, 0], X_test[correct, 1], color='green', label='Točno klasificirani')
plt.scatter(X_test[incorrect, 0], X_test[incorrect, 1], color='black', label='Pogrešno klasificirani')

plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Testni podaci: Točno vs Pogrešno klasificirani')
plt.legend(loc='best')
plt.show()