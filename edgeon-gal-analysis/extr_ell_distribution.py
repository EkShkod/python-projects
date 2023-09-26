import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.pyplot import figure


# Parameters
n = 1000
e = 0.875

# Open table
file_path = '../data/txt/ps1cand2_sex2_all_new.txt'
data = pd.read_table(file_path, sep="\s+\|*", header=[0], skiprows=[1], skipfooter=2, engine='python')
data = data.drop(columns=['e_a', 'e_b', 'radkron', 'fluxauto', 'e_fluxauto', 'magauto', 'e_magauto', 'radpetro',
                          'fluxpetro', 'e_fluxpetro', 'magpetro', 'e_magpetro'])
data.dropna(axis='columns', how='all', inplace=True)

# Creating gri table
gri = ['g', 'r', 'i']
data_gri = data[data['band'].isin(gri)]

# Computing simple ell median
df = data.groupby(['id']).agg({'ell': ['median']}).reset_index()
df.columns = ['_'.join(col).rstrip('_') for col in df.columns.values]
data = data.merge(df, on='id', how='left')

df_gri = data_gri.groupby(['id']).agg({'ell': ['median']}).reset_index()
df_gri.columns = ['_'.join(col).rstrip('_') for col in df_gri.columns.values]
data_gri = data_gri.merge(df, on='id', how='left')

# Remove duplicates
data_gri = data_gri.drop_duplicates(subset=['id'])
data = data.drop_duplicates(subset=['id'])

# Extract the thinnest galaxies with ell > e (optional)
ell_gri = np.array(data_gri['ell_median'])
ell_all = np.array(data['ell_median'])
extr_gri = np.array(data_gri['ell_median'][data_gri['ell_median'] >= e])
extr_all = np.array(data['ell_median'][data['ell_median'] >= e])
extr_gri_id = np.array(data_gri['id'][data_gri['ell_median'] >= e])
extr_all_id = np.array(data['id'][data['ell_median'] >= e])

# Plots
figure(figsize=(10, 8), dpi=100)
plt.hist(ell_gri, bins=50, log=True, color='gray', edgecolor='darkgray', alpha=0.3)
plt.axvline(np.sort(ell_gri)[-n], linestyle='dashed', color='darkgray')
plt.title('Распределение медианного значения ell по фильтрам g, r, i', fontsize=12)
plt.xlabel(r'$ell_{gri}$', fontsize=20)
plt.show()

figure(figsize=(10, 8), dpi=100)
plt.hist(ell_all, bins=50, log=True, color='gray', edgecolor='darkgray', alpha=0.3)
plt.axvline(np.sort(ell_all)[-n], linestyle='dashed', color='darkgray')
plt.title('Распределение медианного значения ell по всем фильтрам', fontsize=12)
plt.xlabel(r'$ell_{all}$', fontsize=20)
plt.show()

# Save lists
file = open(f'extr_ell_id_gri.txt', 'w')
file.write('id' + '\n')
for i in extr_gri_id:
    file.write(str(i) + '\n')
file.close()

file = open(f'extr_ell_id_all.txt', 'w')
file.write('id' + '\n')
for i in extr_all_id:
    file.write(str(i) + '\n')
file.close()
