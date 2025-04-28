import numpy as np
from tensorflow import keras
from keras import layers
from keras import models
from tensorflow.keras.models import load_model
from PIL import Image
import matplotlib.pyplot as plt

# Učitavanje modela
model = load_model("model.keras")

# Učitavanje slike i priprema
img = Image.open("test.png").convert("L").resize((28, 28))
img_arr = np.array(img).astype("float32") / 255
img_arr = np.expand_dims(img_arr, axis=-1)
img_arr = np.expand_dims(img_arr, axis=0)

# Predikcija
pred = model.predict(img_arr)
predicted_class = np.argmax(pred)

# Ispis
plt.imshow(img, cmap="gray")
plt.title(f"Predikcija: {predicted_class}")
plt.axis("off")
plt.show()

print(f"Model predviđa da je broj: {predicted_class}")
print(f"Verovatnoća: {pred[0][predicted_class]:.2f}")
print(f"Verovatnoće za sve klase: {pred[0]}")