#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 22 15:05:45 2021

@author: zachz
"""

#%% Imports

import numpy as np
import matplotlib.pyplot as plt

# Load in .spydata - there is no way to do this with a command

plt.style.use('seaborn-notebook')

#%% Start building plots

n,x,_ = plt.hist(faraut, bins = 10**np.arange(0,4,0.1), histtype=u'step' ,weights=np.zeros_like(faraut) + 1/len(faraut))
bin_centers = 0.5*(x[1:]+x[:-1])
plt.plot(bin_centers,n,'o-',label='Faraut (human)')
plt.axvline(x=faraut_mean,color='tab:orange')

n,x,_ = plt.hist(meg, bins = 10**np.arange(0,4,0.1), histtype=u'step' ,weights=np.zeros_like(meg) + 1/len(meg))
bin_centers = 0.5*(x[1:]+x[:-1])
plt.plot(bin_centers,n,'o-',label='Meg (monkey)')
plt.axvline(meg_mean,color='tab:red')

n,x,_ = plt.hist(minxha, bins = 10**np.arange(0,4,0.1), histtype=u'step' ,weights=np.zeros_like(minxha) + 1/len(minxha))
bin_centers = 0.5*(x[1:]+x[:-1])
plt.plot(bin_centers,n,'o-',label='Minxha (human)')
plt.axvline(minxha_mean,color='tab:brown')

n,x,_ = plt.hist(stein, bins = 10**np.arange(0,4,0.1), histtype=u'step' ,weights=np.zeros_like(stein) + 1/len(stein))
bin_centers = 0.5*(x[1:]+x[:-1])
plt.plot(bin_centers,n,'o-',label='Steinmetz (mouse)')
plt.axvline(stein_mean,color='tab:gray')

plt.xlabel('tau (ms)')
plt.ylabel('proportion')
plt.xscale('log')
plt.title('Amygdala')
plt.legend()
plt.show()
