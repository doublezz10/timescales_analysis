#%% Load data
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import statsmodels.formula.api as smf

from scipy import interpolate

plt.style.use('seaborn')

# AP : ~50 anterior // 15 posterior
# VD = close to 0 is Ventral, 30 is dorsal
# ML = -5 is medial and -25 is lateral

#%%

amyg = pd.read_csv('/Users/zachz/Documents/timescales_analysis/1000iter results/isi figs/fred_amyg_isi.csv')

# %%

ys = amyg.AP.to_numpy()
xs = amyg.ML.to_numpy()
zs = amyg.tau.to_numpy()

avg = amyg.groupby(['VD']).mean() # dimension not plotted here
avg['count'] = amyg.groupby(['VD']).count().unit.values

points = np.column_stack((xs,ys))

grid_xn, grid_yn = np.mgrid[np.min(xs):np.max(xs):20j, np.min(ys):np.max(ys):20j]

f = interpolate.LinearNDInterpolator(points, zs)
zn = f(grid_xn, grid_yn)

plt.pcolor(grid_xn,grid_yn,zn,shading='auto',cmap='Greys',vmin=0,vmax=1000)
plt.xlabel('ML')
plt.ylabel('AP')

plt.colorbar(label='timescale (ms)')

plt.scatter(x=avg['ML'],y=avg['AP'],c=avg['tau'],s=avg['count'],cmap='inferno',vmin=0,vmax=1000)

plt.colorbar()

plt.title('amygdala')

plt.show()

#%%

model = smf.ols('tau ~ AP * ML',data=avg)

res = model.fit()

print(res.summary())

# %%

ys = amyg.AP.to_numpy()
xs = amyg.VD.to_numpy()
zs = amyg.tau.to_numpy()

avg = amyg.groupby(['ML']).mean() # dimension not plotted here
avg['count'] = amyg.groupby(['ML']).count().unit.values

points = np.column_stack((xs,ys))

grid_xn, grid_yn = np.mgrid[np.min(xs):np.max(xs):20j, np.min(ys):np.max(ys):20j]

f = interpolate.LinearNDInterpolator(points, zs)
zn = f(grid_xn, grid_yn)

plt.pcolor(grid_xn,grid_yn,zn,shading='auto',cmap='Greys',vmin=0,vmax=1000)
plt.xlabel('DV')
plt.ylabel('AP')

plt.colorbar(label='timescale (ms)')

plt.scatter(x=avg['VD'],y=avg['AP'],c=avg['tau'],s=avg['count'],cmap='inferno',vmin=0,vmax=1000)

plt.colorbar()

plt.title('amygdala')

plt.show()

# %% 

model = smf.ols('tau ~ AP * VD',data=avg)

res = model.fit()

print(res.summary())

#%%

xs = amyg.ML.to_numpy()
ys = amyg.VD.to_numpy()
zs = amyg.tau.to_numpy()

avg = amyg.groupby(['AP']).mean() # dimension not plotted here
avg['count'] = amyg.groupby(['AP']).count().unit.values

points = np.column_stack((xs,ys))

grid_xn, grid_yn = np.mgrid[np.min(xs):np.max(xs):20j, np.min(ys):np.max(ys):20j]

f = interpolate.LinearNDInterpolator(points, zs)
zn = f(grid_xn, grid_yn)

plt.pcolor(grid_xn,grid_yn,zn,shading='auto',cmap='Greys',vmin=0,vmax=1000)
plt.xlabel('ML')
plt.ylabel('DV')

plt.colorbar(label='timescale (ms)')

plt.scatter(x=avg['ML'],y=avg['VD'],c=avg['tau'],s=avg['count'],cmap='inferno',vmin=0,vmax=1000)

plt.colorbar()

plt.title('amygdala')

plt.show()

#%%

model = smf.ols('tau ~ ML * VD',data=avg)

res = model.fit()

print(res.summary())