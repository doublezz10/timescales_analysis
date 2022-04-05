#%% Load data
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import statsmodels.formula.api as smf

from scipy import interpolate

plt.rcParams['font.size'] = '7'

plt.style.use('seaborn')

# AP : ~50 anterior // 15 posterior
# VD = close to 0 is Ventral, 30 is dorsal
# ML = -5 is medial and -25 is lateral

#%%

#%% Load in data

ofc = pd.read_csv('fred_ofc_isi.csv')
lai = pd.read_csv('fred_lai_isi.csv')
vl = pd.read_csv('fred_vl_isi.csv')

ofc_lai_vl = pd.concat((ofc,lai,vl),ignore_index=True)

ofc_lai_vl['brain_area'] = ofc_lai_vl['brain_area'].str.replace('LAI','AI')
ofc_lai_vl['specific_area'] = ofc_lai_vl['specific_area'].str.replace('LAI','AI')

listofspecies = ['mouse','monkey','human']
ofc_lai_vl['species'] = pd.Categorical(ofc_lai_vl['species'], categories=listofspecies, ordered=True)

# %%

ofc_lai_vl = ofc_lai_vl[ofc_lai_vl.specific_area != '45']
ofc_lai_vl = ofc_lai_vl[ofc_lai_vl.specific_area != '13b']

ofc_lai_vl = np.around(ofc_lai_vl,0)

ys = ofc_lai_vl.AP.to_numpy()
xs = ofc_lai_vl.ML.to_numpy()
zs = ofc_lai_vl.tau.to_numpy()

grouped_df = ofc_lai_vl.groupby(['AP', 'ML'],as_index=False
                          ).mean()

grouped_df['count'] = ofc_lai_vl.groupby(['AP','ML']).count().unit.values

avg = ofc_lai_vl

points = np.column_stack((xs,ys))

grid_xn, grid_yn = np.mgrid[np.min(xs):np.max(xs):32j, np.min(ys):np.max(ys):26j]

f = interpolate.LinearNDInterpolator(points, zs)
zn = f(grid_xn, grid_yn)

plt.figure(figsize=(5,4))

plt.pcolor(grid_xn,grid_yn,zn,shading='auto',cmap='Greys',vmin=0,vmax=1000)
plt.xlabel('ML (mm)',fontsize=7)
plt.ylabel('AP (mm)',fontsize=7)
plt.xticks(fontsize=7)
plt.yticks(fontsize=7)

cbar = plt.colorbar()
cbar.set_label(label='timescale (ms)',size=7)
cbar.ax.set_yticklabels([0,200,400,600,800,1000],fontsize=7)

sns.scatterplot(data=grouped_df,x='ML',y='AP',hue='tau',size='count',palette='viridis',hue_norm=(0,1000),legend=True)

plt.legend(bbox_to_anchor=(-0.2, 1.0),loc='upper right',prop={'size':7})

# sns.scatterplot(x='ML',y='AP',hue='specific_area',data=ofc_lai_vl,alpha=0.2)
# plt.legend(bbox_to_anchor=(-0.5, 1.0), loc='upper left',prop={'size':7})

plt.gca().invert_xaxis()

plt.show()

#%%

model = smf.ols('tau ~ AP * ML',data=avg)

res = model.fit()

print(res.summary())

# %%

ofc_lai_vl = ofc_lai_vl[ofc_lai_vl.specific_area != '45']

ofc_lai_vl = np.around(ofc_lai_vl,0)

ys = ofc_lai_vl.AP.to_numpy()
xs = ofc_lai_vl.ML.to_numpy()
zs = ofc_lai_vl.lat.to_numpy()

grouped_df = ofc_lai_vl.groupby(['AP', 'ML'],as_index=False
                          ).mean()

grouped_df['count'] = ofc_lai_vl.groupby(['AP','ML']).count().unit.values

avg = ofc_lai_vl

points = np.column_stack((xs,ys))

grid_xn, grid_yn = np.mgrid[np.min(xs):np.max(xs):20j, np.min(ys):np.max(ys):20j]

f = interpolate.LinearNDInterpolator(points, zs)
zn = f(grid_xn, grid_yn)

plt.figure(figsize=(5,4))

plt.pcolor(grid_xn,grid_yn,zn,shading='auto',cmap='Greys',vmin=0,vmax=1000)
plt.xlabel('ML (mm)',fontsize=7)
plt.ylabel('AP (mm)',fontsize=7)
plt.xticks(fontsize=7)
plt.yticks(fontsize=7)

cbar = plt.colorbar()
cbar.set_label(label='timescale (ms)',size=7)
cbar.ax.set_yticklabels([0,200,400,600,800,1000],fontsize=7)

sns.scatterplot(data=grouped_df,x='ML',y='AP',hue='lat',size='count',palette='inferno',hue_norm=(0,200),legend=True)

plt.legend(bbox_to_anchor=(-0.2, 1.0),loc='upper right',prop={'size':7})

# sns.scatterplot(x='ML',y='AP',hue='specific_area',data=ofc_lai_vl,alpha=0.2)
# plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')

plt.gca().invert_xaxis()

plt.show()

#%%

model = smf.ols('lat ~ AP * ML',data=avg)

res = model.fit()

print(res.summary())

#%%