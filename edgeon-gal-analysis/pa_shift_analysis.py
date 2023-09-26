import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.pyplot import figure


def pa_compute(pa):
    pa = np.array(pa)

    alpha_median = np.median(pa)
    alpha_std = np.std(pa)

    pa_new = pa + 90
    pa_new[pa_new > 180] -= 180

    beta_median = np.median(pa_new)
    beta_std = np.std(pa_new)

    beta_median -= 90
    if beta_median < 0:
        beta_median += 180

    if np.abs(beta_std) <= np.abs(alpha_std):
        true_median = beta_median
    else:
        true_median = alpha_median
    return true_median


# Parameters
f = 'y'     # Filter name
size = 3    # Min size of galaxies included in sample


# Open and extract data
# Photometric
file_path = '../data/txt/ps1cand2_sex2.txt'
data_sex = pd.read_table(file_path, sep="\s+\|*", header=[0], skiprows=[1], skipfooter=2, engine='python')
data_sex = data_sex.drop(columns=['e_a', 'e_b', 'radkron', 'fluxauto', 'e_fluxauto', 'magauto', 'e_magauto', 'radpetro',
                          'fluxpetro', 'e_fluxpetro', 'magpetro', 'e_magpetro'])
data_sex.dropna(axis='columns', how='all', inplace=True)

# List of excluded objects
file_path_ex = '../data/txt/excluded.txt'
data_ex = pd.read_table(file_path_ex, sep="\s+\|*", header=[0], skiprows=[1], skipfooter=2, engine='python')
data_ex.dropna(axis='columns', how='all', inplace=True)
id_ex = np.array(data_ex['id'])

# List of trashed objects
file_path_trash = '../data/txt/trash.txt'
data_trash = pd.read_table(file_path_trash, header=[0], engine='python')
id_trash = np.array(data_trash['id'])

# Searching for excluded galaxies in dataframes
id_sex = np.array(data_sex['id'])
id_off = np.hstack([id_trash, id_ex])
id_clear = np.setdiff1d(id_sex, id_off)

# Updating dataframes (excluding trash and filtering galaxies by size)
data_sex = data_sex[data_sex['id'].isin(id_clear)]
data_sex = data_sex[data_sex['a'] > size]

# Computing simple PA median
df_1 = data_sex.groupby(['id']).agg({'pa': ['median']}).reset_index()
df_1.columns = ['_'.join(col).rstrip('_') for col in df_1.columns.values]
df_1.rename(columns={'pa_median': 'pa_median_simple'}, inplace=True)

# Computing alternative (fixed) PA median
df_2 = data_sex.groupby(['id'])['pa'].agg([pa_compute]).reset_index()
df_2.rename(columns={'pa_compute': 'pa_median_true'}, inplace=True)

# Inserting computed values into original dataframe
data_sex = data_sex.merge(df_1, on='id', how='left')
data_sex = data_sex.merge(df_2, on='id', how='left')

# Computing and plotting PA shifts for filter f
pa_f = np.array(data_sex['pa'][data_sex['band'] == f])
pa_median_true = np.array(data_sex['pa_median_true'][data_sex['band'] == f])
pa_median_simple = np.array(data_sex['pa_median_simple'][data_sex['band'] == f])
pa_shift_true = np.abs(pa_median_true - pa_f)
pa_shift_simple = np.abs(pa_median_simple - pa_f)

pa_shift_copy = -pa_shift_true
pa_shift = np.hstack([pa_shift_copy, pa_shift_true])
pa_shift = pa_shift[(pa_shift >= -100) & (pa_shift <= 100)]
pa_shift_std = np.std(pa_shift)

pa_shift_sigma = pa_shift_true[pa_shift_true <= 3*pa_shift_std]

figure(figsize=(10, 8), dpi=100)
plt.hist(pa_shift_true, bins=50, log=True, color='gray', edgecolor='darkgray', alpha=0.3, label=f'$a > {size}$')
plt.axvline(3*pa_shift_std, linestyle='dashed', color='darkgary')
plt.title(fr'Сдвиг значений $PA_{f}$ относительно истинного медианного значения $PA_m$', fontsize=15)
plt.xlabel(r'$PA_{shift}$', fontsize=20)
plt.legend()
plt.show()

figure(figsize=(10, 8), dpi=100)
plt.hist(pa_shift_true[pa_shift_true < 100], bins=50, log=True, color='gray', label=f'$a > {size}$')
plt.axvline(3*pa_shift_std, linestyle='dashed', color='darkgray')
plt.title(fr'Сдвиг значений $PA_{f}$ относительно истинного медианного значения $PA_m$', fontsize=15)
plt.xlabel(r'$PA_{shift}$', fontsize=20)
plt.legend()
plt.show()

figure(figsize=(10, 8), dpi=100)
plt.hist(pa_shift_sigma, bins=50, log=True, color='gray', edgecolor='darkgray', alpha=0.3, label=f'$a > {size}$')
plt.title(fr'Сдвиг значений $PA_{f}$ относительно истинного медианного значения $PA_m$', fontsize=15)
plt.xlabel(r'$PA_{shift}$', fontsize=20)
plt.legend()
plt.show()

