import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from matplotlib import pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping


# ucitaj CIFAR-10 podatkovni skup
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# prikazi 9 slika iz skupa za ucenje
plt.figure()
for i in range(9):
    plt.subplot(330 + 1 + i)
    plt.xticks([]),plt.yticks([]) # ne prikazuj oznake osa
    plt.imshow(X_train[i]) # prikazi sliku
plt.show()


# pripremi podatke (skaliraj ih na raspon [0,1]])
X_train_n = X_train.astype('float32')/ 255.0
X_test_n = X_test.astype('float32')/ 255.0

# 1-od-K kodiranje
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
y_train = y_train.astype('uint8')
y_test = y_test.astype('uint8')

# CNN mreza
model = keras.Sequential()

# Ulazni sloj - slike 32x32 piksela sa 3 kanala (RGB)
model.add(layers.Input(shape=(32, 32, 3)))

# Konvolucijski sloj sa 32 filtra, veličina 3x3, ReLU aktivacija
model.add(layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same'))
# Maksimalno sažimanje (max pooling) 2x2
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
# Dropout sloj (isključivanje 25% neurona)
model.add(layers.Dropout(0.25))

# Drugi konvolucijski sloj sa 64 filtra
model.add(layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same'))
# Max pooling
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
# Dropout
model.add(layers.Dropout(0.25))

# Treći konvolucijski sloj sa 128 filtara
model.add(layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same'))
# Max pooling
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
# Dropout
model.add(layers.Dropout(0.25))

# Ravnanje (Flatten) - pretvara volumen u 1D vektor
model.add(layers.Flatten())
# Potpuno povezani sloj sa 500 neurona i ReLU aktivacijom
model.add(layers.Dense(500, activation='relu'))
# Dropout nakon Dense sloja
model.add(layers.Dropout(0.5))
# Izlazni sloj s 10 neurona (10 klasa), softmax aktivacija
model.add(layers.Dense(10, activation='softmax'))

# Ispis arhitekture mreže
model.summary()

# Definiraj funkcije povratnog poziva: TensorBoard + EarlyStopping
my_callbacks = [
    keras.callbacks.TensorBoard(log_dir='logs/cnn_dropout_earlystop', update_freq=100),
    keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=1)
]

# Kompajliraj model (ponovno)
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Pokreni učenje
model.fit(
    X_train_n,
    y_train,
    epochs=40,             # Maksimalno 40 epoha, ali može stati i prije
    batch_size=64,
    callbacks=my_callbacks,
    validation_split=0.1
)

# evaluiraj model
score = model.evaluate(X_test_n, y_test, verbose=0)
print(f'Tocnost na testnom skupu podataka: {100.0*score[1]:.2f}')