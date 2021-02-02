#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 29 13:49:48 2021

@author: zachz
"""

#%% Imports

import numpy as np
import matplotlib.pyplot as plt

# Load in .spydata - there is no way to do this with a command

plt.style.use('seaborn-notebook')
    
#%% Murray et al style plot

plt.plot(human_sorted_mean_taus,'o-')
plt.xticks(range(0,6),human_sorted_brain_areas,rotation=90)
plt.xlabel('brain region')
plt.ylabel('tau (ms)')
plt.title('Human')
plt.show()

#%% Stack bar chart with n_neurons behind line graph

fig, ax1 = plt.subplots()
ax2 = ax1.twinx()  # set up the 2nd axis

ax1.plot(human_sorted_mean_taus,'o-')
ax1.set_ylabel('timescale (ms)')
ax1.set_title('Human')

ax2.bar(range(0,6),(len(faraut_amyg_taus),len(minxha_hc_taus),len(minxha_amyg_taus),len(faraut_hc_taus),len(minxha_presma_taus),len(minxha_dacc_taus)),alpha=0.2)
ax2.set_xticks(range(0,6))
ax2.set_xticklabels(human_sorted_brain_areas,rotation=90)
ax2.grid(b=False)
ax2.set_ylabel('n units')