#%% imports

import scipy.io as spio
import numpy as np
import matplotlib.pyplot as plt
from scipy.io.matlab.mio5 import _simplify_cells
import seaborn as sns
import pandas as pd

import bottleneck as bn

# load different data

amyg_mat = spio.loadmat('/Users/zachz/Dropbox/Timescales across species/Spiketimes only/Chandravadia - Human MTL/chandravadia_amyg_2.mat',simplify_cells=True)

hc_mat = spio.loadmat('/Users/zachz/Dropbox/Timescales across species/Spiketimes only/Chandravadia - Human MTL/chandravadia_hc_2.mat',simplify_cells=True)

amyg_units = amyg_mat['spikes']
hc_units = hc_mat['spikes']

#%% start with hc only bc fewer units so quicker
# data is in seconds

#for unit in range(len(hc_units)):

for unit in range(10):
    
    this_unit = hc_units[unit]
    
    # zero align
    
    this_unit = this_unit - this_unit[0]
    
    # bin data at 500ms
    
    bins = np.arange(0,np.max(this_unit),step=0.05)
    
    binned_spikes, edges = np.histogram(this_unit,bins=bins)
    
    mean_fr = np.sum(binned_spikes)/np.max(this_unit)
    
    # sliding window across 1 sec
    
    slide = bn.move_sum(binned_spikes, window=4) * 5
    
    z_slide = (slide - mean_fr) / mean_fr

    plt.plot(z_slide)
    plt.ylabel('z-scored FR')
    plt.xlabel('n windows')  
    
    plt.show()
    
# %% need a metric for 'selectivity'

# something like n windows > some z-score value
