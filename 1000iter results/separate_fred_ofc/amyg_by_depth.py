#%%

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns

from scipy.stats import linregress, pearsonr

from scipy.io import loadmat

import statsmodels.formula.api as smf

plt.style.use('seaborn')

#%%

amyg = pd.read_csv('fred_amyg.csv')

#%%

plt.scatter(x=amyg.zach_tau,y=amyg.depth,color='blue',label='iterative_fit')
plt.scatter(x=amyg.fred_tau,y=amyg.depth,color='red',label='ISI_fit')

plt.ylabel('depth (mm)')
plt.xlabel('tau (ms)')

plt.title('Morbier amygdala')

plt.legend()

plt.show
# %%

model = smf.ols('fred_lat ~ depth',data=amyg)

res = model.fit()

print(res.summary())


#%%

s = sns.lmplot(data=amyg,x='fred_tau',y='depth',line_kws={'color': '#C1B37F'},scatter_kws={'color': '#C1B37F','s':6,'alpha':0.7},height=3,aspect=1)

s.set_axis_labels(x_var='timescale (ms)',y_var='recording depth (mm)',fontsize=7)
s.set(xlim=(0,1000))
s.set_yticklabels(size = 7)
s.set_xticklabels([0,200,400,600,800,1000],size = 7)
plt.show()

s = sns.lmplot(data=amyg,x='fred_lat',y='depth',line_kws={'color': '#C1B37F'},scatter_kws={'color': '#C1B37F','s':6,'alpha':0.7},height=3,aspect=1,legend=True)

s.set_axis_labels(x_var='latency (ms)',y_var='recording depth (mm)',fontsize=7)
s.set(xlim=(0,200))
s.set_yticklabels(size = 7)
s.set_xticklabels([0,50,100,150,200],size = 7)

plt.show()
# %%

