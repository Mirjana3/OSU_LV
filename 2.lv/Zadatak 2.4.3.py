import numpy as np
import matplotlib.pyplot as plt

img = plt.imread("road.jpg")

brighter = np.clip(img*1.5, 0, 1) # clip - ograničava vrijednost [0,1]

h, w = img.shape[:2] # dimenzija slike, ali uzima se samo prva dva elementa visina i širina
quarter = img[:, w//4:w//2] # uzimaju se svi redovi, dio slike od četvrtine širine (w//4) do polovice širine (w//2) - // cijeli broj dijeljenja

rotated = np.rot90(img, -1) # -1 obrnuto od kazaljske sata

mirrored = np.fliplr(img) # izvrće sliku s lijeva na desno

fig, axs = plt.subplots(2, 2, figsize=(10, 8))
axs[0, 0].imshow(brighter, cmap='gray')
axs[0, 0].set_title('Posvijetljena slika')
axs[0, 1].imshow(quarter, cmap='gray')
axs[0, 1].set_title('Druga četvrtina slike')
axs[1, 0].imshow(rotated, cmap='gray')
axs[1, 0].set_title('Rotirana slika')
axs[1, 1].imshow(mirrored, cmap='gray')
axs[1, 1].set_title('Zrcaljena slika')

plt.show()
