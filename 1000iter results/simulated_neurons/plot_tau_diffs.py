#%%

import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

tau_diffs = pd.read_csv('/Users/zachz/Documents/timescales_analysis/1000iter results/simulated_neurons/tau_diffs.csv')

plt.style.use('seaborn')

#%%

plt.figure(figsize=(11,8.5))

sns.scatterplot(data=tau_diffs,x='true_tau',y='iter_tau_diff')

plt.plot(range(300),range(300),linestyle='dashed')
    
plt.xlabel('ground_truth tau')
plt.ylabel('iter_tau')
    
plt.show()
# %%
