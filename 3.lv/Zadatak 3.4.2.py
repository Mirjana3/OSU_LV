from matplotlib.colors import ListedColormap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('data_C02_emission.csv') # učitavanje csv datoteke

# emisijaCO2 na histogramu
plt.figure()
plt.hist(data['CO2 Emissions (g/km)'])
plt.title('Histogram emisije CO2')
plt.grid(True)
plt.show()

# odnos gradske potrošnje goriva i emisije CO2
plt.figure()
cmap = ListedColormap(['blue', 'orange', 'green', 'red'])
scatter = plt.scatter(data['Fuel Consumption City (L/100km)'], 
                      data['CO2 Emissions (g/km)'], 
                      c = pd.Categorical(data['Fuel Type']).codes, 
                      cmap=cmap)
plt.xlabel('Gradska potrošnja goriva')
plt.ylabel('CO2 emisija')
plt.title('Odnos gradske potrošnje goriva sa emisijom CO2')
fuel_types = data['Fuel Type'].unique()
handles = [plt.Line2D([0], [0], marker='o', color='w', label=fuel_type, 
                      markerfacecolor=cmap(i)) for i, fuel_type in enumerate(fuel_types)]
plt.legend(handles=handles, title="Vrsta goriva")
plt.show()

# razdioba izvangradske potrošnje s obzirom na tip goriva
data.boxplot(column= 'Fuel Consumption Hwy (L/100km)', by='Fuel Type', patch_artist=True)
plt.xlabel('Tip goriva')
plt.ylabel('Izvangradska potrošnja (L/100km)')
plt.title('Razdioba izvangradske potrošnje goriva')
plt.suptitle("")
plt.show()

# broj vozila po tipu goriva
plt.figure()
gorivo = data.groupby('Fuel Type').size()
gorivo.plot(kind='bar')
plt.xlabel('Tip goriva')
plt.ylabel('Broj vozila')
plt.title('Broj vozila po tipu goriva')
plt.xticks(rotation=0)
plt.show()

# prosječnu CO2 emisiju vozila s obzirom na broj cilindara
cilindri = data.groupby('Cylinders')['CO2 Emissions (g/km)'].mean()
plt.figure()
cilindri.plot(kind='bar')
plt.xlabel('Broj cilindara')
plt.ylabel('Prosječna CO2 emisija (g/km)')
plt.title('Prosječna CO2 emisija vozila s obzirom na broj cilindara')
plt.xticks(rotation=0)
plt.show()