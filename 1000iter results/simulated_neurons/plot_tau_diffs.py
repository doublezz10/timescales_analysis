#%%

import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

tau_diffs = pd.read_csv('/Users/zachz/Documents/timescales_analysis/1000iter results/simulated_neurons/one_vs_iter.csv')
tau_diffs2 = pd.read_csv('/Users/zachz/Documents/timescales_analysis/1000iter results/simulated_neurons/tau_diffs.csv')

plt.style.use('seaborn')

#%%

fig, axs = plt.subplots(1,2,figsize=(11,8.5))

sns.scatterplot(ax = axs[0],data=tau_diffs,x='true_tau',y='iter_tau')

axs[0].plot(range(200),range(200),linestyle='dashed')
    
axs[0].set_xlabel('ground truth tau (ms)')
axs[0].set_ylabel('iteratively fit tau (ms)')
    
sns.scatterplot(ax=axs[1],data=tau_diffs,x='true_tau',y='one_fit')

axs[1].plot(range(200),range(200),linestyle='dashed')
    
axs[1].set_xlabel('ground truth tau (ms)')
axs[1].set_ylabel('one fit tau (ms)')
    
    
plt.tight_layout()

plt.show()
# %%
