#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 29 14:50:35 2021

@author: zachz
"""

#%% Imports

import matplotlib.pyplot as plt

# Load in .spydata - there is no way to do this with a command

#%% Start plotting

fig, axs = plt.subplots(2,2)

ax0 = axs[0,0].twinx()  # set up the 2nd axis
ax1 = axs[0,1].twinx()
ax2 = axs[1,0].twinx()
ax3 = axs[1,1].twinx()

axs[0,0].plot(mouse_sorted_mean_taus,'o-')
axs[0,0].set_ylabel('timescale (ms)')
axs[0,0].set_title('Mouse')
axs[0,0].set_ylim([0,650])

ax0.bar(range(0,9),(len(ca2_taus),len(dg_taus),len(ca1_taus),len(ca3_taus),len(orb_taus),len(aca_taus),len(pl_taus),len(bla_taus),len(ila_taus)),alpha=0.2)
ax0.set_xticks(range(0,9))
ax0.set_xticklabels(mouse_sorted_brain_areas)
ax0.grid(b=False)
ax0.set_ylim([0,1000])
ax0.set_yticks([])

axs[0,1].plot(rat_sorted_mean_taus,'o-')
axs[0,1].set_title('Rat')
axs[0,1].set_ylim([0,650])
axs[0,1].set_yticks([])

ax1.bar(range(0,3),(len(buzsaki_ofc_taus),len(buzsaki_acc_taus),len(buzsaki_mpfc_taus)),alpha=0.2)
ax1.set_xticks(range(0,3))
ax1.set_xticklabels(rat_sorted_brain_areas)
ax1.grid(b=False)
ax1.set_ylabel('n units')
ax1.set_ylim([0,1000])

axs[1,0].plot(monkey_sorted_mean_taus,'o-')
axs[1,0].set_ylabel('timescale (ms)')
axs[1,0].set_title('Monkey')
axs[1,0].set_ylim([0,650])

ax2.bar(range(0,7),(len(meg_sc_taus),len(froot_ofc_post_taus),len(froot_acc_pre_taus),len(froot_ofc_post_taus),len(meg_amyg_taus),len(meg_vs_taus),len(froot_acc_post_taus)),alpha=0.2)
ax2.set_xticks(range(0,7))
ax2.set_xticklabels(list(('Meg \n scACC','FROOT \n OFC post','FROOT \n ACC pre','FROOT \n OFC post','Meg \n Amyg','Meg \n vStriatum','FROOT \n ACC post')))
ax2.grid(b=False)
ax2.set_ylim([0,1000])
ax2.set_yticks([])

axs[1,1].plot(human_sorted_mean_taus,'o-')
axs[1,1].set_title('Human')
axs[1,1].set_ylim([0,650])
axs[1,1].set_yticks([])

ax3.bar(range(0,6),(len(faraut_amyg_taus),len(minxha_hc_taus),len(minxha_amyg_taus),len(faraut_hc_taus),len(minxha_presma_taus),len(minxha_dacc_taus)),alpha=0.2)
ax3.set_xticks(range(0,6))
ax3.set_xticklabels(list(('Faraut \n Amyg','Minxha \n HC','Minxha \n Amyg','Faraut \n HC','Minxha \n preSMA','Minxha \n dACC')))
ax3.grid(b=False)
ax3.set_ylabel('n units')
ax3.set_ylim([0,1000])

plt.tight_layout()
plt.show()