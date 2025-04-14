import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import dendrogram
from sklearn.datasets import make_blobs, make_circles, make_moons
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.cluster import KMeans

def generate_data(n_samples, flagc):
    # 3 grupe
    if flagc == 1:
        random_state = 365
        X,y = make_blobs(n_samples=n_samples, random_state=random_state)
    
    # 3 grupe
    elif flagc == 2:
        random_state = 148
        X,y = make_blobs(n_samples=n_samples, random_state=random_state)
        transformation = [[0.60834549, -0.63667341], [-0.40887718, 0.85253229]]
        X = np.dot(X, transformation)

    # 4 grupe 
    elif flagc == 3:
        random_state = 148
        X, y = make_blobs(n_samples=n_samples,
                        centers = 4,
                        cluster_std=np.array([1.0, 2.5, 0.5, 3.0]),
                        random_state=random_state)
    # 2 grupe
    elif flagc == 4:
        X, y = make_circles(n_samples=n_samples, factor=.5, noise=.05)
    
    # 2 grupe  
    elif flagc == 5:
        X, y = make_moons(n_samples=n_samples, noise=.05)
    
    else:
        X = []
        
    return X

# generiranje podatkovnih primjera
X = generate_data(500, 1)

# prikazi primjere u obliku dijagrama rasprsenja
plt.figure()
plt.scatter(X[:,0],X[:,1])
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.title('podatkovni primjeri')
plt.show()


# 2. Zadatak
# primjena K-means algoritma
K = 3 
kmeans = KMeans(n_clusters=K, n_init=10, random_state=0)
labels = kmeans.fit_predict(X)

# prikaz rezultata grupiranja
plt.figure()
plt.scatter(X[:,0], X[:,1], c=labels)
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.title(f'Grupirani podatkovni primjeri za K={K}')
plt.show()

# 3. Zadatak
# mjenjanje flagc i K 
for flagc in range(1, 6):
    X = generate_data(500, flagc)
    
    if flagc in [1,2]:
        K = 3
    elif flagc in [3, 4, 5]:
        K = 4

    kmeans = KMeans(n_clusters=K, n_init=10, random_state=0)
    labels = kmeans.fit_predict(X)

    plt.figure()
    plt.scatter(X[:, 0], X[:, 1], c=labels)
    plt.title(f'flagc = {flagc}, K = {K}')
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    plt.show()

# elbow method
sse = []
for k in range(1,10):
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=0)
    kmeans.fit(X)
    sse.append(kmeans.inertia_)

plt.plot(range(1,10), sse, marker='o')
plt.xticks(range(1,10))
plt.xlabel('Broj grupa (K)')    
plt.ylabel('SSE')
plt.title('Lakat metoda')
plt.grid(True)
plt.show()
