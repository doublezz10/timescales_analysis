#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 28 15:35:52 2021

@author: zachz
"""

#%% Imports

import numpy as np
import matplotlib.pyplot as plt

# Load in .spydata - there is no way to do this with a command

plt.style.use('seaborn-notebook')

#%% Start building plots

n,x,_ = plt.hist(ofc_pre_taus, bins = 10**np.arange(0,4,0.1), histtype=u'step' ,weights=np.zeros_like(ofc_pre_taus) + 1/len(ofc_pre_taus))
bin_centers = 0.5*(x[1:]+x[:-1])
plt.plot(bin_centers,n,'o-',label='OFC pre')
plt.axvline(x=ofc_pre_mean,color='tab:orange')

n,x,_ = plt.hist(acc_pre_taus, bins = 10**np.arange(0,4,0.1), histtype=u'step' ,weights=np.zeros_like(acc_pre_taus) + 1/len(acc_pre_taus))
bin_centers = 0.5*(x[1:]+x[:-1])
plt.plot(bin_centers,n,'o-',label='ACC pre')
plt.axvline(x=acc_pre_mean,color='tab:red')

n,x,_ = plt.hist(ofc_post_taus, bins = 10**np.arange(0,4,0.1), histtype=u'step' ,weights=np.zeros_like(ofc_post_taus) + 1/len(ofc_post_taus))
bin_centers = 0.5*(x[1:]+x[:-1])
plt.plot(bin_centers,n,'o-',label='OFC post')
plt.axvline(x=ofc_post_mean,color='tab:brown')

n,x,_ = plt.hist(acc_post_taus, bins = 10**np.arange(0,4,0.1), histtype=u'step' ,weights=np.zeros_like(acc_post_taus) + 1/len(acc_post_taus))
bin_centers = 0.5*(x[1:]+x[:-1])
plt.plot(bin_centers,n,'o-',label='ACC post')
plt.axvline(x=acc_post_mean,color='tab:grey')

plt.xlabel('tau (ms)')
plt.ylabel('proportion')
plt.xscale('log')
plt.title('FROOT')
plt.legend()
plt.show()