import numpy as np
import matplotlib.pyplot as plt

x = np.array([1, 3, 3, 2, 1])  
y = np.array([1, 1, 2, 2, 1])      

plt.plot(x, y, 'b', linewidth=1, marker='.', markersize=10)

plt.axis([0, 4, 0, 4])
plt.xlabel('X osa')
plt.ylabel('Y osa')
plt.title('Zadatak 2.4.1')
plt.legend()
plt.grid(True)

plt.show()