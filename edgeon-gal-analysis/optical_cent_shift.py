import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.pyplot import figure


# Parameters
f = 'i'     # Filter name
size = 5.5  # Min size of galaxies included in sample


# Open and extract data
# Photometric
file_path = '../data/txt/ps1cand2_sex2.txt'
data = pd.read_table(file_path, sep="\s+\|*", header=[0], skiprows=[1], skipfooter=2, engine='python')
data = data.drop(columns=['e_a', 'e_b', 'radkron', 'fluxauto', 'e_fluxauto', 'magauto', 'e_magauto', 'radpetro',
                          'fluxpetro', 'e_fluxpetro', 'magpetro', 'e_magpetro'])
data.dropna(axis='columns', how='all', inplace=True)

# Mean parameters
file_path = '../data/txt/ps1cand2_sex2_stats_all.txt'
data_all = pd.read_table(file_path, sep="\s+\|*", header=[0], skiprows=[1], skipfooter=2, engine='python')
data_all.dropna(axis='columns', how='all', inplace=True)

# List of excluded objects
file_path_ex = '../data/txt/excluded.txt'
data_ex = pd.read_table(file_path_ex, sep="\s+\|*", header=[0], skiprows=[1], skipfooter=2, engine='python')
data_ex.dropna(axis='columns', how='all', inplace=True)

# Searching for excluded galaxies in dataframes
id_excluded = np.array(data_ex['id'])
id_sex = np.array(data['id'])
id_all = np.array(data_all['id'])
id_clear = np.setdiff1d(id_sex, id_excluded)
id_clean = np.setdiff1d(id_all, id_excluded)

# Updating dataframes
data = data[data['id'].isin(id_clear)]
data_all = data_all[data_all['id'].isin(id_clean)]

# Extracting parameters
ra_f = np.array(data['ra'][data['band'] == f])
dec_f = np.array(data['dec'][data['band'] == f])
a_f = np.array(data['a'][data['band'] == f])

id_dec = np.array(data['id'][data['dec'].notna()][data['band'] == f])
id_ra = np.array(data['id'][data['ra'].notna()][data['band'] == f])
id_a = np.array(data['id'][data['a'].notna()][data['band'] == f])
id_com = np.intersect1d(id_dec, id_ra)
id = list(np.intersect1d(id_com, id_a))

data_all = data_all[data_all['id'].isin(id)]
id = np.array(data_all['id'])
ra_mean = np.array(data_all['ra_mean'])
dec_mean = np.array(data_all['dec_mean'])
ra_median = np.array(data_all['ra_median'])
dec_median = np.array(data_all['dec_median'])
a_mean = np.array(data_all['a_mean'])
a_median = np.array(data_all['a_median'])

# Computing optical center shift
s = np.arccos(np.sin(dec_f) * np.sin(dec_median) + np.cos(dec_f) * np.cos(dec_median) * np.cos(ra_f - ra_median))

# Extracting parameters for galaxies with a > 5.5
data_1 = data[data['a'] > size]
ra_f_1 = np.array(data_1['ra'][data_1['band'] == f])
dec_f_1 = np.array(data_1['dec'][data_1['band'] == f])
a_f_1 = np.array(data_1['a'][data_1['band'] == f])

id_dec_1 = np.array(data_1['id'][data_1['dec'].notna()][data_1['band'] == f])
id_ra_1 = np.array(data_1['id'][data_1['ra'].notna()][data_1['band'] == f])
id_a_1 = np.array(data_1['id'][data_1['a'].notna()][data_1['band'] == f])
id_com_1 = np.intersect1d(id_dec_1, id_ra_1)
id_1 = list(np.intersect1d(id_com_1, id_a_1))

data_all_1 = data_all[data_all['id'].isin(id_1)]
id_1 = np.array(data_all_1['id'])
ra_mean_1 = np.array(data_all_1['ra_mean'])
dec_mean_1 = np.array(data_all_1['dec_mean'])
ra_median_1 = np.array(data_all_1['ra_median'])
dec_median_1 = np.array(data_all_1['dec_median'])
a_mean_1 = np.array(data_all_1['a_mean'])
a_median_1 = np.array(data_all_1['a_median'])

# Computing optical center shift for galaxies with a > 5.5
s_1 = np.arccos(np.sin(dec_f_1) * np.sin(dec_median_1) + np.cos(dec_f_1)
                * np.cos(dec_median_1) * np.cos(ra_f_1 - ra_median_1))

# Histogram
figure(figsize=(10, 8), dpi=100)
plt.hist(np.abs(s) / a_median, bins=50, log=True, color='gray', edgecolor='darkgray', alpha=0.3, label='all')
plt.hist(np.abs(s_1) / a_median_1, bins=50, log=True, color='steelblue', edgecolor='darkblue', alpha=0.3, label=f'$a > {size}$')
plt.title(fr'Сдвиг оптического центра ($S_{f}$) относительно размера галактики ($a_{f}$)', fontsize=15)
plt.xlabel(r'$|\frac{S}{a_{median}}|$', fontsize=20)
plt.legend()
plt.show()

# IDs of interesting galaxies
par = np.abs(s) / a_median
id_tail = id[(par >= 0.0025) & (par <= 0.005)]

file = open("s_i_tail_0025_005_id.txt", 'w')
file.write('id' + '\n')
for i in id_tail:
    file.write(str(i) + '\n')
file.close()

