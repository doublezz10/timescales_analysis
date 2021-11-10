#%%

import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

tau_diffs = pd.read_csv('/Users/zachz/Documents/timescales_analysis/1000iter results/simulated_neurons/one_vs_iter.csv')
tau_diffs = pd.read_csv('/Users/zachz/Documents/timescales_analysis/1000iter results/simulated_neurons/tau_diffs.csv')

plt.style.use('seaborn')

#%%

plt.figure(figsize=(11,8.5))

sns.scatterplot(data=tau_diffs[:20],x='true_tau',y='iter_tau')

plt.plot(range(200),range(200),linestyle='dashed')
    
plt.xlabel('ground truth tau (ms)')
plt.ylabel('iteratively fit tau (ms)')
    
plt.show()
# %%

plt.figure(figsize=(11,8.5))

sns.scatterplot(data=tau_diffs[:20],x='true_tau',y='one_fit')

plt.plot(range(200),range(200),linestyle='dashed')
    
plt.xlabel('ground truth tau (ms)')
plt.ylabel('one fit tau (ms)')
    
plt.show()
# %%
