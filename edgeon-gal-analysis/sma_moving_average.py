import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.pyplot import figure
import copy


# Parameters
n = 300         # Size of moving average window
thr = 75        # sma threshold to exclude outliers
stp = 1        # sma window step


# Open table and collect data
file_path = '../data/txt/ps1cand2_sex2_stats_gri.txt'
data = pd.read_table(file_path, sep="\s+\|*", header=[0], skiprows=[1], skipfooter=2, engine='python')
data.dropna(axis='columns', how='all', inplace=True)

file_path_0 = '../data/txt/ps1cand2.txt'
data_0 = pd.read_table(file_path_0, sep="\s+\|*", header=[0], skiprows=[1], skipfooter=2, engine='python')
data_0.dropna(axis='columns', how='all', inplace=True)

id_sex = np.array(data['id'])
id_sma = np.array(data_0['id'][data_0['sma'] < thr])
id_int = np.intersect1d(id_sma, id_sex)

data_0 = data_0[data_0['id'].isin(id_int)]
data = data[data['id'].isin(id_int)]

sma = np.array(data_0['sma'])
a_mean = np.array(data['a_mean'])
a_median = np.array(data['a_median'])

# Calculate moving average
sma_copy = copy.deepcopy(sma)
mean_dict = dict(zip(a_mean, sma))
mean_dict = dict(sorted(mean_dict.items(), key=lambda x: x[0]))
a_mean = np.array(list(mean_dict.keys()))
sma = np.array(list(mean_dict.values()))

med_dict = dict(zip(a_median, sma_copy))
med_dict = dict(sorted(med_dict.items(), key=lambda x: x[0]))
a_median = np.array(list(med_dict.keys()))
sma_copy = np.array(list(med_dict.values()))

mov_sma = pd.Series(sma).rolling(window=n, step=stp).mean().iloc[n-1:].values
mov_sma_copy = pd.Series(sma_copy).rolling(window=n, step=stp).mean().iloc[n-1:].values
mov_a_mean = pd.Series(a_mean).rolling(window=n, step=stp).mean().iloc[n-1:].values
mov_a_median = pd.Series(a_median).rolling(window=n, step=stp).mean().iloc[n-1:].values

figure(figsize=(10, 8), dpi=100)
plt.scatter(a_mean, sma, color='gray', s=2)
plt.scatter(mov_a_mean, mov_sma, marker='.', color='steelblue')
plt.title(r'Зависимость параметра sma от среднего значения a', fontsize=15)
plt.ylabel(r'sma', fontsize=15)
plt.xlabel(r'$a_{mean}$', fontsize=15)
plt.semilogx()
plt.semilogy()
plt.show()

figure(figsize=(10, 8), dpi=100)
plt.scatter(a_median, sma_copy, color='gray', s=2)
plt.scatter(mov_a_median, mov_sma_copy, marker='.', color='steelblue')
plt.title(r'Зависимость параметра sma от медианного значения a', fontsize=15)
plt.ylabel(r'sma', fontsize=15)
plt.xlabel(r'$a_{median}$', fontsize=15)
plt.semilogx()
plt.semilogy()
plt.show()

