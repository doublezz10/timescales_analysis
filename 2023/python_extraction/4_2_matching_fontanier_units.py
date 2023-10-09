#%%

# imports

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

new_data = pd.read_csv('processed_data.csv')
old_data = pd.read_csv('/Users/zachz/Documents/timescales_analysis/2023/font_mcc_results.csv')
# old_data = pd.read_csv('/Users/zachz/Documents/timescales_analysis/2023/mcc_python_isi.csv')


#%%

# get matches

match_data = old_data.join(new_data[new_data.dataset=='fontanier'].set_index('unit'),on='unit',lsuffix='_old',rsuffix='_new',how='outer')

#%%

# filter match data

good_data = []

for entry in range(len(match_data)):
    
    this_cell = match_data.iloc[entry]
    
    if np.logical_and(this_cell.tau_new > 10, this_cell.tau_new < 1000) and np.logical_and(this_cell.tau_old > 10, this_cell.tau_old < 1000):
        
        good_data.append(this_cell)
        
good_data = pd.DataFrame(good_data,columns=['tau_old', 'tau_error_old', 'A_old', 'B_old',
       'fit_err_old', 'dof_old', 'fit_old', 'rmse_old', 'peak_lat_old',
       'peak_old', 'dip_lat_old', 'dip_old', 'peak_second_lat_old',
       'peak_second_old', 'unit_old', 'tau_new', 'tau_error_new', 'A_new',
       'B_new', 'fit_err_new', 'dof_new', 'fit_new', 'rmse_new',
       'peak_lat_new', 'peak_new', 'dip_lat_new', 'dip_new',
       'peak_second_lat_new', 'peak_second_new', 'nbSpk', 'unit_new',
       'dataset', 'area', 'fr', 'species', 'brain_region'])

#%%

# do correlations

sns.lmplot(data=match_data,x='tau_old',y='tau_new')
plt.show()

# %%
