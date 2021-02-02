#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 22 16:02:32 2021

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
plt.axvline(x=buzsaki_mean,color='tab:orange')

n,x,_ = plt.hist(hunt, bins = 10**np.arange(0,4,0.1), histtype=u'step' ,weights=np.zeros_like(hunt) + 1/len(hunt))
bin_centers = 0.5*(x[1:]+x[:-1])
plt.plot(bin_centers,n,'o-',label='Hunt (monkey)')

n,x,_ = plt.hist(meg, bins = 10**np.arange(0,4,0.1), histtype=u'step' ,weights=np.zeros_like(meg) + 1/len(meg))
bin_centers = 0.5*(x[1:]+x[:-1])
plt.plot(bin_centers,n,'o-',label='Meg (monkey)')
plt.axvline(x=meg_mean,color='tab:brown')

n,x,_ = plt.hist(minxha, bins = 10**np.arange(0,4,0.1), histtype=u'step' ,weights=np.zeros_like(minxha) + 1/len(minxha))
bin_centers = 0.5*(x[1:]+x[:-1])
plt.plot(bin_centers,n,'o-',label='Minxha (human)')
plt.axvline(x=minxha_mean,color='tab:grey')

n,x,_ = plt.hist(stein, bins = 10**np.arange(0,4,0.1), histtype=u'step' ,weights=np.zeros_like(stein) + 1/len(stein))
bin_centers = 0.5*(x[1:]+x[:-1])
plt.plot(bin_centers,n,'o-',label='Steinmetz (mouse)')
plt.axvline(x=stein_mean,color='tab:cyan')

plt.xlabel('tau (ms)')
plt.ylabel('proportion')
plt.xscale('log')
plt.title('ACC')
plt.legend()
plt.show()
