import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt('data.csv', delimiter=',', skiprows=1)  

broj_osoba = data.shape[0]
print(f'Broj osoba u skupu podataka: {broj_osoba}')

# min, max i srednja visina
min_visina = np.min(data[:, 1])
max_visina = np.max(data[:, 1])
srednja_visina = np.mean(data[:, 1])
print(f'Minimalna visina: {min_visina:.2f} cm')
print(f'Maksimalna visina: {max_visina:.2f} cm')
print(f'Srednja visina: {srednja_visina:.2f} cm')

# muškarci u odnosu na žene
muskarci = data[data[:, 0] == 1]
zene = data[data[:, 0] == 0]

print(f'Srednja visina muškaraca: {np.mean(muskarci[:, 1]):.2f} cm')
print(f'Srednja visina žena: {np.mean(zene[:, 1]):.2f} cm')

fig, axs = plt.subplots(1, 2, figsize=(12, 6)) 

# odnos visine i mase
axs[0].scatter(data[:, 1], data[:, 2], alpha=0.5, label='Svi podaci')
axs[0].set_xlabel('Visina (cm)')
axs[0].set_ylabel('Masa (kg)')
axs[0].set_title('Odnos visine i mase')
axs[0].grid(True)
axs[0].legend()

# odnos visine i mase za svaku 50. osoba
axs[1].scatter(data[::50, 1], data[::50, 2], color='red', alpha=0.7, label='Svaka 50. osoba')
axs[1].set_xlabel('Visina (cm)')
axs[1].set_ylabel('Masa (kg)')
axs[1].set_title('Odnos visine i mase za svaku 50. osobu')
axs[1].grid(True)
axs[1].legend()

plt.tight_layout()
plt.show()