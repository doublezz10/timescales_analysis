#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 22 16:23:16 2021

@author: zachz
"""

#%% Imports

import numpy as np
import matplotlib.pyplot as plt

# Load in .spydata - there is no way to do this with a command

plt.style.use('seaborn-notebook')

#%% Start building plots

n,x,_ = plt.hist(buzsaki, bins = 10**np.arange(0,4,0.1), histtype=u'step' ,weights=np.zeros_like(buzsaki) + 1/len(buzsaki))
bin_centers = 0.5*(x[1:]+x[:-1])
plt.plot(bin_centers,n,'o-',label='Buzsaki (rat)')

n,x,_ = plt.hist(minxha, bins = 10**np.arange(0,4,0.1), histtype=u'step' ,weights=np.zeros_like(minxha) + 1/len(minxha))
bin_centers = 0.5*(x[1:]+x[:-1])
plt.plot(bin_centers,n,'o-',label='Minxha (human)')

n,x,_ = plt.hist(stein_ila, bins = 10**np.arange(0,4,0.1), histtype=u'step' ,weights=np.zeros_like(stein_ila) + 1/len(stein_ila))
bin_centers = 0.5*(x[1:]+x[:-1])
plt.plot(bin_centers,n,'o-',label='Steinmetz ILA (mouse)')

n,x,_ = plt.hist(stein_pl, bins = 10**np.arange(0,4,0.1), histtype=u'step' ,weights=np.zeros_like(stein_pl) + 1/len(stein_pl))
bin_centers = 0.5*(x[1:]+x[:-1])
plt.plot(bin_centers,n,'o-',label='Steinmetz PL (mouse)')


plt.xlabel('tau (ms)')
plt.ylabel('proportion')
plt.xscale('log')
plt.title('Medial Frontal Cortex')
plt.legend()
plt.show()