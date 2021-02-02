#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 29 11:57:10 2021

@author: zachz
"""

#%% Imports

import numpy as np
import matplotlib.pyplot as plt

# Load in .spydata - there is no way to do this with a command

plt.style.use('seaborn-notebook')

#%% Murray et al style plot

plt.figure(figsize=(5,5))

plt.plot(monkey_sorted_mean_taus,'o-')
plt.xticks(range(0,7),monkey_sorted_brain_areas,rotation=90)
plt.xlabel('brain region')
plt.ylabel('tau (ms)')
plt.title('Monkey')
plt.show()

#%% Stack bar chart with n_neurons behind line graph

fig, ax1 = plt.subplots()
ax2 = ax1.twinx()  # set up the 2nd axis

ax1.plot(monkey_sorted_mean_taus,'o-')
ax1.set_ylabel('timescale (ms)')
ax1.set_title('Monkey')

ax2.bar(range(0,7),(len(meg_sc_taus),len(froot_ofc_post_taus),len(froot_acc_pre_taus),len(froot_ofc_post_taus),len(meg_amyg_taus),len(meg_vs_taus),len(froot_acc_post_taus)),alpha=0.2)
ax2.set_xticks(range(0,7))
ax2.set_xticklabels(list(('Meg scACC','FROOT \n OFC post','FROOT \n ACC pre','FROOT \n OFC post','Meg Amyg','Meg vStriatum','FROOT \n ACC post')))
ax2.tick_params(axis='x', rotation=90)
ax2.grid(b=False)
ax2.set_ylabel('n units')

plt.show()

#%% Polar plot

fig = plt.figure(figsize=(7,7))
ax = fig.add_subplot(111,polar=True)
theta = [102.5,115,127.5,152.5,177.5,255,315]
#            ACC * 3       OFC * 2  VS  Amyg
theta = np.radians(np.array(theta))
r = [froot_acc_pre,froot_acc_post,meg_sc,froot_ofc_pre,froot_ofc_post,meg_vs,meg_amyg]
c = ax.plot(theta,r,'o-')
labels = np.radians(np.array([45,115,165,215,255,315]))
bars = np.radians(np.array([0,90,140,190,240,270]))
bar_width = np.radians(np.array([90,50,50,50,30,90]))
ax.set_xticks(labels)
ax.set_xticklabels(list(('HC','ACC','OFC','other FC','Striatum','Amygdala')))
for bar in range(len(labels)):
    ax.bar(labels[bar],400,width=bar_width[bar],alpha=0.1)
plt.title('Monkey')
label_position=ax.get_rlabel_position()
ax.text(np.radians(label_position+10),ax.get_rmax()/2.,'tau (ms)',
        rotation=label_position,ha='center',va='center')
