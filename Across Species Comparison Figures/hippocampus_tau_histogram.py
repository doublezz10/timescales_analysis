#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 22 16:47:48 2021

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

n,x,_ = plt.hist(minxha, bins = 10**np.arange(0,4,0.1), histtype=u'step' ,weights=np.zeros_like(minxha) + 1/len(minxha))
bin_centers = 0.5*(x[1:]+x[:-1])
plt.plot(bin_centers,n,'o-',label='Minxha (human)')

n,x,_ = plt.hist(stein, bins = 10**np.arange(0,4,0.1), histtype=u'step' ,weights=np.zeros_like(stein) + 1/len(stein))
bin_centers = 0.5*(x[1:]+x[:-1])
plt.plot(bin_centers,n,'o-',label='Steinmetz - all subfields (mouse)')

plt.xlabel('tau (ms)')
plt.ylabel('proportion')
plt.xscale('log')
plt.title('Hippocampus')
plt.legend()
plt.show()