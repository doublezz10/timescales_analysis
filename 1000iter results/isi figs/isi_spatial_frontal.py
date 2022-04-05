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

# ofc_lai_vl = np.around(ofc_lai_vl,2)

ys = ofc_lai_vl.AP.to_numpy()
xs = ofc_lai_vl.ML.to_numpy()
zs = ofc_lai_vl.tau.to_numpy()

avg = ofc_lai_vl.groupby(['VD']).mean() # dimension not plotted here
avg['count'] = ofc_lai_vl.groupby(['VD']).count().unit.values

points = np.column_stack((xs,ys))

grid_xn, grid_yn = np.mgrid[np.min(xs):np.max(xs):20j, np.min(ys):np.max(ys):20j]

f = interpolate.LinearNDInterpolator(points, zs)
zn = f(grid_xn, grid_yn)

plt.pcolor(grid_xn,grid_yn,zn,shading='auto',cmap='Greys',vmin=0,vmax=1000)
plt.xlabel('ML')
plt.ylabel('AP')

#plt.colorbar(label='timescale (ms) interpolated')

# plt.scatter(x=avg['ML'],y=avg['AP'],c=avg['tau'],s=avg['count'],cmap='inferno',vmin=0,vmax=1000,alpha=0.3)
# cbar2 = plt.colorbar()
# cbar2.ax.set_yticklabels([])  

sns.scatterplot(x='ML',y='AP',hue='specific_area',data=ofc_lai_vl,alpha=0.2)
plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')

plt.title('Frontal cortex')

plt.show()

#%%

model = smf.ols('tau ~ AP * ML',data=avg)

res = model.fit()

print(res.summary())

# %%

ys = ofc_lai_vl.AP.to_numpy()
xs = ofc_lai_vl.VD.to_numpy()
zs = ofc_lai_vl.tau.to_numpy()

avg = ofc_lai_vl.groupby(['ML']).mean() # dimension not plotted here
avg['count'] = ofc_lai_vl.groupby(['ML']).count().unit.values

points = np.column_stack((xs,ys))

grid_xn, grid_yn = np.mgrid[np.min(xs):np.max(xs):20j, np.min(ys):np.max(ys):20j]

f = interpolate.LinearNDInterpolator(points, zs)
zn = f(grid_xn, grid_yn)

plt.pcolor(grid_xn,grid_yn,zn,shading='auto',cmap='Greys',vmin=0,vmax=1000)
plt.xlabel('DV')
plt.ylabel('AP')

plt.colorbar(label='timescale (ms) interpolated')

plt.scatter(x=avg['VD'],y=avg['AP'],c=avg['tau'],s=avg['count'],cmap='inferno',vmin=0,vmax=1000)
cbar2 = plt.colorbar()
cbar2.ax.set_yticklabels([])  

# sns.scatterplot(x='VD',y='AP',hue='specific_area',data=ofc_lai_vl,alpha=0.2)
# plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')

plt.title('Frontal cortex')

plt.tight_layout()

plt.show()

#%%

model = smf.ols('tau ~ AP * VD',data=avg)

res = model.fit()

print(res.summary())

#%%

xs = ofc_lai_vl.ML.to_numpy()
ys = ofc_lai_vl.VD.to_numpy()
zs = ofc_lai_vl.tau.to_numpy()

avg = ofc_lai_vl.groupby(['AP']).mean() # dimension not plotted here
avg['count'] = ofc_lai_vl.groupby(['AP']).count().unit.values

points = np.column_stack((xs,ys))

grid_xn, grid_yn = np.mgrid[np.min(xs):np.max(xs):20j, np.min(ys):np.max(ys):20j]

f = interpolate.LinearNDInterpolator(points, zs)
zn = f(grid_xn, grid_yn)

plt.pcolor(grid_xn,grid_yn,zn,shading='auto',cmap='Greys',vmin=0,vmax=1000)
plt.xlabel('ML')
plt.ylabel('DV')

cbar1 = plt.colorbar(label='timescale (ms) interpolated',orientation='horizontal')

# plt.scatter(x=avg['ML'],y=avg['VD'],c=avg['tau'],s=avg['count'],cmap='inferno',vmin=0,vmax=1000)
# cbar2 = plt.colorbar()
# cbar2.ax.set_yticklabels([])  

sns.scatterplot(x='ML',y='VD',hue='specific_area',data=ofc_lai_vl,alpha=0.05,palette='viridis')
plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')

plt.title('Frontal cortex')

#plt.tight_layout()

plt.show()

#%%

model = smf.ols('tau ~ ML * VD',data=avg)

res = model.fit()

print(res.summary())

# %%
