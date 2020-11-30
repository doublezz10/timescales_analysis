#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 12:13:58 2020

@author: zachz
"""
#%% Imports

import numpy as np
import matplotlib.pyplot as plt

#%% Subplots of curve fits

fig, axs = plt.subplots(2, 2, sharex=True, sharey=True,figsize=(8,6))

axs[0,0].errorbar(x_m,minxha_amyg_mean,yerr=minxha_amyg_se,label='original data')
axs[0,0].plot(x_m,func(x_m,*minxha_amyg_pars),label='fit curve')
# axs[0,0].legend(loc='upper right')
axs[0,0].set_title('amygdala')
axs[0,0].text(600,.07,'tau = %i' %minxha_amyg_pars[1])

axs[0,1].errorbar(x_m,minxha_dacc_mean,yerr=minxha_dacc_se,label='original data')
axs[0,1].plot(x_m,func(x_m,*minxha_dacc_pars),label='fit curve')
# axs[0,1].legend(loc='upper right')
axs[0,1].set_title('dACC')
axs[0,1].text(600,.07,'tau = %i' %minxha_dacc_pars[1])

axs[1,0].errorbar(x_m,minxha_hc_mean,yerr=minxha_hc_se,label='original data')
axs[1,0].plot(x_m,func(x_m,*minxha_hc_pars),label='fit curve')
# axs[1,0].legend(loc='upper right')
axs[1,0].set_title('hippocampus')
axs[1,0].text(600,.07,'tau = %i' %minxha_hc_pars[1])

axs[1,1].errorbar(x_m,minxha_presma_mean,yerr=minxha_presma_se,label='original data')
axs[1,1].plot(x_m,func(x_m,*minxha_presma_pars),label='fit curve')
# axs[1,1].legend(loc='upper right')
axs[1,1].set_title('preSMA')
axs[1,1].text(600,.07,'tau = %i' %minxha_presma_pars[1])

fig.add_subplot(111,frame_on=False)
plt.tick_params(labelcolor="none", bottom=False, left=False)

plt.xlabel("lag (ms)")
plt.ylabel("autocorrelation")
plt.title('Minxha human single units')

plt.show()

#%% Subplots of tau histograms

fig, axs = plt.subplots(2, 2, sharex=True, sharey=True,figsize=(8,6))

axs[0,0].hist(np.log(minxha_amyg_taus))
axs[0,0].set_title('amygdala')
axs[0,0].text(-10,175,'n_units = %i' %len(minxha_amyg_taus))

axs[0,1].hist(np.log(minxha_dacc_taus))
axs[0,1].set_title('dACC')
axs[0,1].text(-10,175,'n_units = %i' %len(minxha_dacc_taus))

axs[1,0].hist(np.log(minxha_hc_taus))
axs[1,0].set_title('hippocampus')
axs[1,0].text(-10,150,'n_units = %i' %len(minxha_hc_taus))

axs[1,1].hist(np.log(minxha_presma_taus))
axs[1,1].set_title('preSMA')
axs[1,1].text(-10,150,'n_units = %i' %len(minxha_presma_taus))

fig.add_subplot(111,frame_on=False)
plt.tick_params(labelcolor="none", bottom=False, left=False)

plt.xlabel("log(tau)")
plt.ylabel("count")
plt.title('Minxha human single units')

plt.show()
