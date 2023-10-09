#%%

# imports

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#%%

data = pd.read_csv('fontanier_10iter.csv')
data.iter = np.hstack((np.ones(298),2*np.ones(298),3*np.ones(298),4*np.ones(298),5*np.ones(298),6*np.ones(298),7*np.ones(298),8*np.ones(298),9*np.ones(298),10*np.ones(298)))

# %%

# plot distributions acorss iters

sns.histplot(data=data,x='tau',hue='iter',element='step',fill=None)
plt.show()

# %%

# plot spreads across units

sns.violinplot(data=data,x='unit',y='tau')
plt.show()

# %%
