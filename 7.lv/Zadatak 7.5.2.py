import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as Image
from sklearn.cluster import KMeans
from pathlib import Path
from PIL import Image as PILImage

# ucitaj sliku
img = Image.imread("imgs\\test_1.jpg")

# prikazi originalnu sliku
plt.figure()
plt.title("Originalna slika")
plt.imshow(img)
plt.tight_layout()
plt.show()

# pretvori vrijednosti elemenata slike u raspon 0 do 1
img = img.astype(np.float64) / 255

# transfromiraj sliku u 2D numpy polje (jedan red su RGB komponente elementa slike)
w,h,d = img.shape
img_array = np.reshape(img, (w*h, d))

# rezultatna slika
img_array_aprox = img_array.copy()

# 1. Zadatak
# Broj različitih boja u slici
n_unique_colors = len(np.unique(img_array, axis=0))
print(f"Broj različitih boja na slici: {n_unique_colors}")

# 2. Zadatak
# Primjena K-means algoritma
K = 5
kmeans = KMeans(n_clusters=K, n_init=10, random_state=0)
kmeans.fit(img_array)
labels = kmeans.predict(img_array)

# 3. Zadatak
# Zamjena boja s centroidima
img_array_aprox = kmeans.cluster_centers_[labels]
img_array_aprox = np.reshape(img_array_aprox, (w, h, d))

# 4. Zadatak
# Prikaz kvantizirane slike
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.title("Originalna slika")
plt.imshow(img)
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title(f"Kvantizirana slika s K={K}")
plt.imshow(img_array_aprox)
plt.axis('off')

plt.tight_layout()
plt.show()

# 5. Zadatak
image_folder = Path("imgs")
image_files = [f"test_{i}.jpg" for i in range(1, 7)]

K = 50

for image_name in image_files:
    img_path = image_folder / image_name
    img = np.array(PILImage.open(img_path))
    

    if image_name != "test_4.jpg":
        ima = img.astype(np.float64)
    else:
        ima = img.astype(np.float64) /255
    
    w, h, d = img.shape
    img_array = np.reshape(ima, (w * h, d))
    
    kmeans = KMeans(n_clusters=K, n_init=10, random_state=0)
    labels = kmeans.fit_predict(img_array)
    
    img_aprox = kmeans.cluster_centers_[labels]
    img_aprox = np.reshape(img_aprox, (w, h, d))
    
    if image_name != "test_4.jpg":
        img_aprox_display = img_aprox.astype(np.uint8)
    else:
        img_aprox_display = (img_aprox * 255).astype(np.uint8)
    
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title(f"Original - {image_name}")
    plt.imshow(img)
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.title(f"K-means (K={K}) - {image_name}")
    plt.imshow(img_aprox_display)
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

# 6. Zadatak
# Lakat metoda
inertias = []
K_range = range(1, 11)
for k in K_range:
    km = KMeans(n_clusters=k, n_init=10, random_state=0)
    km.fit(img_array)
    inertias.append(km.inertia_)

plt.figure()
plt.plot(K_range, inertias, marker='o')
plt.title("Lakat metoda")
plt.xlabel("K")
plt.ylabel("J")
plt.grid(True)
plt.show()

# 7. Zadatak
# Binarne slike po grupama
K = 5
for i in range(K):
    mask = (labels == i)
    binary_img = np.zeros_like(img_array)
    binary_img[mask] = 1.0
    binary_img = np.reshape(binary_img, (w, h, d))

    # Spriječi mikroskopske negativne vrijednosti
    binary_img = np.clip(binary_img, 0, 1).astype(np.float32)


    plt.figure()
    plt.title(f"Binarna slika za grupu {i}")
    plt.imshow(binary_img)
    plt.axis('off')
    plt.show()