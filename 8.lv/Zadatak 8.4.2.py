import numpy as np
from tensorflow.keras.models import load_model
from tensorflow import keras
from matplotlib import pyplot as plt

# Učitavanje modela i podataka
model = load_model("model.keras")
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Priprema podataka
x_test_s = x_test.astype("float32") / 255
x_test_s = np.expand_dims(x_test_s, -1)
x_test_s = np.expand_dims(x_test_s, axis=-1)

# Predikcija
y_pred = model.predict(x_test_s)
y_pred_classes = np.argmax(y_pred, axis=1)

# Prikaz loše klasificiranih slika
count = 0
for i in range(len(x_test)):
    if y_pred_classes[i] != y_test[i]:
        plt.subplot(1, 5, count+1)
        plt.imshow(x_test[i], cmap='gray')
        plt.title(f"T: {y_test[i]} / P: {y_pred_classes[i]}")
        plt.axis('off')
        count += 1
        if count == 5:
            break
plt.tight_layout()
plt.show()