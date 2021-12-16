#%%

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns

from scipy.stats import linregress

from scipy.io import loadmat

import statsmodels.formula.api as smf

plt.style.use('seaborn')

#%%

amyg = pd.read_csv('fred_amyg.csv')

#%%

plt.scatter(x=amyg.tau_diff,y=amyg.depth,color='blue',label='iterative_fit')
# plt.scatter(x=amyg.fred_tau,y=amyg.depth,color='red',label='ISI_fit')

plt.ylabel('depth (mm)')
plt.xlabel('tau (ms)')

plt.title('Morbier amygdala')

plt.legend()

plt.show
# %%

model = smf.ols('zach_tau ~ depth',data=amyg)

res = model.fit()

print(res.summary())
# %%
