#%% Load data, calculate mean across iterations
 
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm

plt.style.use('seaborn')

individual_data = pd.read_csv('/Users/zachz/Documents/timescales_analysis/1000iter results/isi figs/all_individuals.csv')
listofspecies = ['mouse','monkey','human']

individual_data['species'] = pd.Categorical(individual_data['species'], categories = listofspecies , ordered = True)

#%%

pvals = []

stein_data = individual_data[individual_data.dataset=='steinmetz']

stein_data["individual"] = stein_data["individual"].astype('category')
stein_data["individual_"] = stein_data["individual"].cat.codes

model = smf.ols(formula='tau ~ brain_area + individual_',data=stein_data)

res = model.fit()

print(res.summary())

pvals.append((res.pvalues['individual_'],'steinmetz','tau'))

#%%

model = smf.ols(formula='lat ~ brain_area + individual_',data=stein_data)

res = model.fit()

print(res.summary())

pvals.append((res.pvalues['individual_'],'steinmetz','lat'))

#%%

chan_data = individual_data[individual_data.dataset=='chandravadia']

chan_data["individual"] = chan_data["individual"].astype('category')
chan_data["individual_"] = chan_data["individual"].cat.codes

model = smf.ols(formula='tau ~ brain_area + individual_',data=chan_data)

res = model.fit()

print(res.summary())

pvals.append((res.pvalues['individual_'],'chandravadia','tau'))

#%%

model = smf.ols(formula='lat ~ brain_area + individual_',data=chan_data)

res = model.fit()

print(res.summary())

pvals.append((res.pvalues['individual_'],'chandravadia','lat'))

pvals = pd.DataFrame(pvals,columns=['pval','dataset','parameter'])

pvals['logp'] = -1*np.log10(pvals['pval'])

# %%

sns.barplot(data=pvals,x='dataset',y='logp',hue='parameter')

plt.axhline(y=-1*np.log10(0.05))

plt.xlabel('')
plt.ylabel('-log(p)')

plt.show()
# %%
