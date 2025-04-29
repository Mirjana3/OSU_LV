import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from matplotlib import pyplot as plt
import tensorboard 

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
model.add(layers.Input(shape=(32,32,3))) # ulazni sloj veličine 32x32x3 (RGB)

model.add(layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')) # 1. konvoucijski sloj 32 dimenzija 3x3
model.add(layers.MaxPooling2D(pool_size=(2, 2))) # 1. sloj za smanjenje dimenzionalnosti (max pooling)

model.add(layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')) # 2. konvoucijski sloj 64 dimenzija 3x3
model.add(layers.MaxPooling2D(pool_size=(2, 2))) # 2. sloj za smanjenje dimenzionalnosti (max pooling)

model.add(layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same')) # 3. konvoucijski sloj 128 dimenzija 3x3
model.add(layers.MaxPooling2D(pool_size=(2, 2))) # 3. sloj za smanjenje dimenzionalnosti (max pooling)

model.add(layers.Flatten()) # sloj za preoblikovanje podataka u 1D vektor
model.add(layers.Dense(500, activation='relu')) # gusto povezani sloj s 500 neurona
model.add(layers.Dense(10, activation='softmax')) # izlazni sloj s 10 neurona (za 10 kl

model.summary() # ispis sažetka modela

# definiraj listu s funkcijama povratnog poziva
my_callbacks = [
    keras.callbacks.TensorBoard(log_dir = 'logs/cnn',
                                update_freq = 100)
]

model.compile(optimizer='adam', # optimizator Adam
                loss='categorical_crossentropy',# funkcija gubitka kategorijska unakrsna entropija
                metrics=['accuracy']) # i metrika točnosti

# treniraj model
model.fit(X_train_n,
            y_train,
            epochs = 40,
            batch_size = 64,
            callbacks = my_callbacks,
            validation_split = 0.1)

# evaluiraj model
score = model.evaluate(X_test_n, y_test, verbose=0)
print(f'Tocnost na testnom skupu podataka: {100.0*score[1]:.2f}')