import numpy as np
import matplotlib.pyplot as plt

size = 50

black = np.zeros((size, size))
white = np.ones((size, size))

top = np.hstack((black, white))
bottom = np.hstack((white, black))
image = np.vstack((top, bottom))

plt.imshow(image, cmap='gray')
plt.title('Crno-bijeli kvadrati')
plt.show()