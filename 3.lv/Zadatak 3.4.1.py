import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('data_C02_emission.csv') # učitavanje csv datoteke

# broj mjerenja
print("Broj mjerenja: ", data.shape[0])
# tip podatka
print("Tipovi podataka: ")
print(data.dtypes.to_string())
# izostale vrijednosti
print("Broj izostalih vrijednosti: ")
print(data.isnull().sum())
# duplicirane vrijednosti
print("Broj duplikata: ", data.duplicated().sum())

#brisanje duplikata i izostalih vrijednosti
data.drop_duplicates(inplace=True)
data.dropna(inplace=True)

# kategoričke vrijednosti konvertiraj u tip category
data[data.select_dtypes(include=['object']).columns] = data.select_dtypes(include=['object']).astype('category')

# tri automobila koja imaju najveću odnosno najmanju gradsku potrošnju
print("Tri vozila s najvećom gradskom potrošnjom: ")
print(data.nlargest(3, 'Fuel Consumption City (L/100km)')[['Make', 'Model', 'Fuel Consumption City (L/100km)']].reset_index(drop=True).to_string(index=False))
print("Tri vozila s najmanjom gradskom potrošnjom: ")
print(data.nsmallest(3, 'Fuel Consumption City (L/100km)')[['Make', 'Model', 'Fuel Consumption City (L/100km)']].reset_index(drop=True).to_string(index=False))

# vozila s veličinom motora između 2.3 i 3.5 L
vozilo = data[(data['Engine Size (L)'] >= 2.5) & (data['Engine Size (L)'] <= 3.5)]
print("Broj vozila s veličinom motora između 2.5 i 3.5: ", vozilo.shape[0])
#prosječna emisija tih vozila
print("Prosječna emisija: ", vozilo['CO2 Emissions (g/km)'].mean())

# broj vozila proivođača audi
vozilo_audi = data[data['Make'] == 'Audi']
print("Broj vozila Audi: ", vozilo_audi.shape[0])
# kolika je prosječna CO2 emisija automobila Audi koji imaju 4 cilindra
audi_4_cilindra = vozilo_audi[vozilo_audi['Cylinders'] == 4]
print("Proječna C02 emisija vozila Audi sa 4 cilindra: ", audi_4_cilindra['CO2 Emissions (g/km)'].mean())

# broj vozila sa 4,6,8,... cilindara
broj_cilindara = [4, 6, 8, 10, 12, 14, 16]
vozilo_cilindri = data[data['Cylinders'].isin(broj_cilindara)]
cilindri = vozilo_cilindri.groupby('Cylinders').size().reset_index(name='Broj vozila')
print("Broj vozila sa 4,6,8,... cilindara: ")
print(cilindri.to_string(index=False))
# prosječna emisija CO2 s obzirom na brojj cilindara
cilindri_emisija = vozilo_cilindri.groupby('Cylinders')['CO2 Emissions (g/km)'].mean().reset_index(name='Proječna emisija CO2')
print("Prosječna emislija CO2 za vozila sa 4, 6, 8, ...: ")
print(cilindri_emisija.to_string(index=False))

# prosječna gradska potrošnja vozila koji koriste dizel
dizel = data[data['Fuel Type'] == 'D']['Fuel Consumption City (L/100km)']
print("Prosječna gradska potrošnja za dizel: ", dizel.mean())
# prosječna gradska potrošnja vozila koji koriste benzin
benzin = data[data['Fuel Type'] == 'X']['Fuel Consumption City (L/100km)']
print("Prosječna gradska potrošnja za benzin: ", benzin.mean())
# medijalne vrijednosti
print("Medijalne vrijednosti dizela: ", dizel.median())
print("Medijalne vrijednosti benzina: ", benzin.median())
 
# vozilo s 4 cilindra i dizelskim motorom ima najveću gradsku potrošnju
dizel_4_cilindra = data[(data['Cylinders'] == 4) & (data['Fuel Type'] == 'D')]
print("Najveću gradsku potrošnju od vozila sa 4 cilindra i dizelskim motorom: ")
print(dizel_4_cilindra.nlargest(1, 'Fuel Consumption City (L/100km)')[['Make', 'Model', 'Fuel Consumption City (L/100km)']].to_string(index=False))

# vozila sa ručnim mjenjačem
ručni_mjenjač = data[data['Transmission'].str.startswith('M')]
print("Broj vozila sa ručnim mjenjačem: ", ručni_mjenjač.shape[0])

# korelacija između numeričkih veličina
print("Korelacija između numeričkih veličina: ")
print(data.corr(numeric_only=True))

# Korelacija između veličine motora i emisije CO2 (0.83)
# To znači da veći motor obično emituje više CO2
# Također se to vidi i sa cilindrima što vozilo ima više
# cilindara to emituje više CO2