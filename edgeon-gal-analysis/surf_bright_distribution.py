import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.pyplot import figure


# Parameters
filters = ['g', 'r', 'i']
a_limits = [3, 5, 7]
mu_lim = 20

for f in filters:
    for a_lim in a_limits:
        # Open tables
        file_path = '../data/txt/ps1cand2_sex2_all_new.txt'
        data = pd.read_table(file_path, sep="\s+\|*", header=[0], skiprows=[1], skipfooter=2, engine='python')
        data = data.drop(columns=['e_a', 'e_b', 'radkron', 'fluxauto', 'e_fluxauto', 'magauto', 'e_magauto', 'radpetro',
                                  'fluxpetro', 'e_fluxpetro', 'e_magpetro'])
        data.dropna(axis='columns', how='all', inplace=True)

        file_path_un = '../data/txt/all_good_id.txt'
        data_un = pd.read_table(file_path_un, header=[0])

        # Collect data
        id_un = np.array(data_un['id'])
        id_f = np.array(data['id'][data['id'].isin(id_un)][data['band'] == f][data['a'] > a_lim])
        a_f = np.array(data['a'][data['id'].isin(id_un)][data['band'] == f][data['a'] > a_lim])
        b_f = np.array(data['b'][data['id'].isin(id_un)][data['band'] == f][data['a'] > a_lim])
        m_f = np.array(data['magpetro'][data['id'].isin(id_un)][data['band'] == f][data['a'] > a_lim])

        # Compute surface brightness
        mu_f = m_f + 2.5 * np.log10(np.pi * a_f * b_f)

        print('f=', f)
        print('a_lim=', a_lim)
        print(len(mu_f[mu_f > mu_lim]))

        # Plot histogram
        figure(figsize=(10, 8), dpi=100)
        plt.hist(mu_f, bins=50, log=True, color='gray', edgecolor='darkgray', alpha=0.3)
        plt.axvline(mu_lim, linestyle='dashed', color='darkgray')
        plt.title(fr'Распределение значений поверхностной яркости $\mu$ в фильтре {f} для a > {a_lim}', fontsize=15)
        plt.xlabel(fr'$\mu_{f}$', fontsize=20)
        plt.savefig(f'sb_{f}_{a_lim}_{mu_lim}')
        
