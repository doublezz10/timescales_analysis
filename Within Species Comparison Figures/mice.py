#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 28 16:25:49 2021

@author: zachz
"""

#%% Imports

import numpy as np
import matplotlib.pyplot as plt

# Load in .spydata - there is no way to do this with a command

plt.style.use('seaborn-notebook')

#%% Murray et al style plot

plt.plot(mouse_sorted_mean_taus,'o-')
plt.xticks(range(0,9),mouse_sorted_brain_areas)
plt.xlabel('brain region')
plt.ylabel('tau (ms)')
plt.title('Mouse')
plt.show()

#%% Stack bar chart with n_neurons behind line graph

fig, ax1 = plt.subplots()
ax2 = ax1.twinx()  # set up the 2nd axis

ax1.plot(mouse_sorted_mean_taus,'o-')
ax1.set_ylabel('timescale (ms)')
ax1.set_title('Mouse')

ax2.bar(range(0,9),(len(ca2_taus),len(dg_taus),len(ca1_taus),len(ca3_taus),len(orb_taus),len(aca_taus),len(pl_taus),len(bla_taus),len(ila_taus)),alpha=0.2)
ax2.set_xticks(range(0,9))
ax2.set_xticklabels(mouse_sorted_brain_areas)
ax2.grid(b=False)
ax2.set_ylabel('n units')