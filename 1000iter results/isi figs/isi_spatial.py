#%% Load data
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm

plt.style.use('seaborn')

# AP : ~50 anterior // 15 posterior
# VD = close to 0 is Ventral, 30 is dorsal
# ML = -5 is medial and -25 is lateral

#%%

fred_data_loc = pd.read_csv('/Users/zachz/Library/CloudStorage/Box-Box/Timescales across species/isi_data_locations.csv')

fred_data_loc = fred_data_loc.rename(columns={'unitID': 'unit', 'name': 'dataset', 'area': 'brain_area'})

fred_data_loc = fred_data_loc[fred_data_loc.keep==1]

fred_data_loc = fred_data_loc[fred_data_loc.species=='monkey']
fred_data_loc = fred_data_loc[fred_data_loc.dataset=='stoll']
# %%
amyg = fred_data_loc[fred_data_loc.brain_area=='AMG']

ofc = fred_data_loc[fred_data_loc.brain_area=='OFC']
ai = fred_data_loc[fred_data_loc.brain_area=='LAI']
vl = fred_data_loc[fred_data_loc.brain_area=='vlPFC']

ai['brain_area'] = 'AI'

ofc_lai_vl = pd.concat((ofc,ai,vl),ignore_index=True)

# %%

from scipy import interpolate

xs = ofc_lai_vl.AP.to_numpy()
ys = ofc_lai_vl.ML.to_numpy()
zs = ofc_lai_vl.tau.to_numpy()

points = np.column_stack((xs,ys))

grid_xn, grid_yn = np.mgrid[np.min(xs):np.max(xs):100j, np.min(ys):np.max(ys):100j]

f = interpolate.LinearNDInterpolator(points, zs)
zn = f(grid_xn, grid_yn)

plt.pcolor(grid_xn,grid_yn,zn,shading='auto',cmap='inferno')
plt.xlabel('AP')
plt.ylabel('ML')
plt.colorbar(label='timescale (ms)')
plt.show()

# %%

xs = ofc_lai_vl.AP.to_numpy()
ys = ofc_lai_vl.VD.to_numpy()
zs = ofc_lai_vl.tau.to_numpy()

points = np.column_stack((xs,ys))

grid_xn, grid_yn = np.mgrid[np.min(xs):np.max(xs):100j, np.min(ys):np.max(ys):100j]

f = interpolate.LinearNDInterpolator(points, zs)
zn = f(grid_xn, grid_yn)

plt.pcolor(grid_xn,grid_yn,zn,shading='auto',cmap='inferno')
plt.xlabel('AP')
plt.ylabel('DV')
plt.colorbar(label='timescale (ms)')
plt.show()

#%%

xs = ofc_lai_vl.ML.to_numpy()
ys = ofc_lai_vl.VD.to_numpy()
zs = ofc_lai_vl.tau.to_numpy()

points = np.column_stack((xs,ys))

grid_xn, grid_yn = np.mgrid[np.min(xs):np.max(xs):100j, np.min(ys):np.max(ys):100j]

f = interpolate.LinearNDInterpolator(points, zs)
zn = f(grid_xn, grid_yn)

plt.pcolormesh(grid_xn,grid_yn,zn,shading='gouraud',cmap='inferno')
plt.xlabel('ML')
plt.ylabel('DV')
plt.colorbar(label='timescale (ms)')
plt.show()

# %%
