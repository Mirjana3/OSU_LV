import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler

labels = {0: 'Adelie', 1: 'Chinstrap', 2: 'Gentoo'}

def plot_decision_regions(X, y, classifier, resolution=0.02):
    plt.figure()
    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    
    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    
    # plot class examples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0],
                    y=X[y == cl, 1],
                    alpha=0.8,
                    c=colors[idx],
                    marker=markers[idx],
                    label=labels[cl])

# ucitaj podatke
df = pd.read_csv("penguins.csv")

# izostale vrijednosti po stupcima
print(df.isnull().sum())

# spol ima 11 izostalih vrijednosti; izbacit cemo ovaj stupac
df = df.drop(columns=['sex'])

# obrisi redove s izostalim vrijednostima
df.dropna(axis=0, inplace=True)

# kategoricka varijabla vrsta - kodiranje
df['species'].replace({'Adelie' : 0, 'Chinstrap' : 1, 'Gentoo': 2}, inplace=True)

print(df.info())

# izlazna velicina: species
output_variable = ['species']

# ulazne velicine: bill length, flipper_length
input_variables = ['bill_length_mm',
                    'flipper_length_mm']

X = df[input_variables].to_numpy()
y = df[output_variable].to_numpy()

# podjela train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 123)

# (a) Prikaz broja primjera za svaku klasu u trening i test skupu
train_class_counts = np.unique(y_train, return_counts=True)
test_class_counts = np.unique(y_test, return_counts=True)


plt.figure(figsize=(10, 6))
# Trening podaci
plt.subplot(1, 2, 1)
plt.bar(train_class_counts[0], train_class_counts[1], color='blue')
plt.title('Broj primjera u trening skupu')
plt.xlabel('Vrsta pingvina')
plt.ylabel('Broj primjera')
# Test podaci
plt.subplot(1, 2, 2)
plt.bar(test_class_counts[0], test_class_counts[1], color='red')
plt.title('Broj primjera u testnom skupu')
plt.xlabel('Vrsta pingvina')
plt.ylabel('Broj primjera')

plt.tight_layout()
plt.show()

# (b) Izgradnja modela logističke regresije
model = LogisticRegression(max_iter=500)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model.fit(X_train, y_train.ravel())  # y_train treba biti jednodimenzionalan niz

# (c) Parametri modela
print("Koeficijenti modela:\n", model.coef_)
print("Intercepti modela:\n", model.intercept_)

# Razlika u odnosu na binarni klasifikacijski problem:
# U višeklasnom problemu logistička regresija koristi jedan model za svaku klasu (takozvani 'one-vs-rest' pristup)
# Svaka klasa ima svoj set koeficijenata, dok u binarnom problemu imamo samo jedan set koeficijenata.

# (d) Prikaz granice odluke za trening podatke
plot_decision_regions(X_train, y_train.ravel(), classifier=model)
plt.xlabel('Duljina kljuna')
plt.ylabel('Duljina peraje')
plt.title('Granica odluke logističke regresije na trening skupu')
plt.legend()
plt.show()

# Komentar: Granica odluke se koristi za vizualizaciju kako model razdvaja različite klase. Ako je granica vrlo zakrivljena, to može ukazivati na složenost podataka.

# (e) Predviđanje za testne podatke
y_pred = model.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
print("Matrica zabune:\n", cm)

accuracy = accuracy_score(y_test, y_pred)
print("Točnost:", accuracy)

report = classification_report(y_test, y_pred, target_names=labels.values())
print("Izvještaj o klasifikaciji:\n", report)

# (f) Dodavanje nove značajke u model
df['body_mass_g'] = df['body_mass_g'].fillna(df['body_mass_g'].mean())  # Popunjavanje izostalih vrijednosti s prosjekom

# Ponovno razdvajanje skupa podataka s novim ulazom
X_new = df[['bill_length_mm', 'flipper_length_mm', 'body_mass_g']]
X_train_new, X_test_new, y_train_new, y_test_new = train_test_split(X_new, y, test_size=0.2, random_state=123)

# Novi model
model_new = LogisticRegression(max_iter=1000)
model_new.fit(X_train_new, y_train_new.ravel())

# Predviđanje i evaluacija novog modela
y_pred_new = model_new.predict(X_test_new)
new_accuracy = accuracy_score(y_test_new, y_pred_new)
print("Nova točnost:", new_accuracy)

# Izvještaj o klasifikaciji s novim ulazima
new_report = classification_report(y_test_new, y_pred_new, target_names=labels.values())
print("Izvještaj o klasifikaciji s dodatnim ulazima:\n", new_report)