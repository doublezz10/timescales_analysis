#%%

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns

from scipy.stats import linregress

import statsmodels.formula.api as smf

from scipy.io import loadmat

plt.style.use('seaborn')

plt.rcParams['font.size'] = '7'


#%%

ofc = pd.read_csv('fred_ofc.csv')

areas = ['11l','11m','13l','13m']

ofc['specific_area'] = pd.Categorical(ofc['specific_area'],categories=areas,ordered=True)

#%%

s = sns.lmplot(data=ofc[ofc.specific_area!='13b'],x='fred_tau',y='depth',hue='specific_area',scatter_kws={'s':6,'alpha':0.7},height=3,aspect=1,legend=False)

s.set_axis_labels(x_var='timescale (ms)',y_var='recording depth (mm)',fontsize=7)
s.set(xlim=(0,1000))
s.set_yticklabels(size = 7)
s.set_xticklabels([0,200,400,600,800,1000],size = 7)

plt.legend(title='',prop={'size': 7})

plt.show()

g = sns.lmplot(data=ofc[ofc.specific_area!='13b'],x='fred_lat',y='depth',hue='specific_area',scatter_kws={'s':6,'alpha':0.7},height=3,aspect=1,legend=False)

g.set_axis_labels(x_var='latency (ms)',y_var='recording depth (mm)',fontsize=7)
g.set(xlim=(0,200))
g.set_yticklabels(size = 7)
g.set_xticklabels([0,50,100,150,200],size = 7)

plt.show()
# %%

model = smf.ols('fred_tau ~ depth + specific_area',data=ofc)

res = model.fit()

print(res.summary())
# %%

model = smf.ols('fred_lat ~ depth + specific_area',data=ofc)

res = model.fit()

print(res.summary())
# %%

for area in areas:
    
    print(area)
    
    try:
    
        model = smf.ols('fred_tau ~ depth',data=ofc[ofc.specific_area == area])

        res = model.fit()  
        
        print('tau p-val = %.3f' %res.pvalues['depth'])
        
        model = smf.ols('fred_lat ~ depth',data=ofc[ofc.specific_area == area])

        res = model.fit() 
        
        print('lat p-val = %.3f' %res.pvalues['depth'])
    
    except:
        
        print('fail')

# %%
