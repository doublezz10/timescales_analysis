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

axs[0,0].errorbar(x_m,mean_amygdala_m,yerr=se_amygdala_m,label='original data')
axs[0,0].plot(x_m,func(x_m,*pars_amy_m),label='fit curve')
# axs[0,0].legend(loc='upper right')
axs[0,0].set_title('amygdala')
axs[0,0].text(600,.07,'tau = %i' %pars_amy_m[1])

axs[0,1].errorbar(x_m,mean_dacc,yerr=se_dacc,label='original data')
axs[0,1].plot(x_m,func(x_m,*pars_dacc),label='fit curve')
# axs[0,1].legend(loc='upper right')
axs[0,1].set_title('dACC')
axs[0,1].text(600,.07,'tau = %i' %pars_dacc[1])

axs[1,0].errorbar(x_m,mean_hc_m,yerr=se_hc_m,label='original data')
axs[1,0].plot(x_m,func(x_m,*pars_hc_m),label='fit curve')
# axs[1,0].legend(loc='upper right')
axs[1,0].set_title('hippocampus')
axs[1,0].text(600,.07,'tau = %i' %pars_hc_m[1])

axs[1,1].errorbar(x_m,mean_presma,yerr=se_presma,label='original data')
axs[1,1].plot(x_m,func(x_m,*pars_presma),label='fit curve')
# axs[1,1].legend(loc='upper right')
axs[1,1].set_title('preSMA')
axs[1,1].text(600,.07,'tau = %i' %pars_presma[1])

fig.add_subplot(111,frame_on=False)
plt.tick_params(labelcolor="none", bottom=False, left=False)

plt.xlabel("lag (ms)")
plt.ylabel("autocorrelation")
plt.title('Minxha human single units')

plt.show()

#%% Subplots of tau histograms

fig, axs = plt.subplots(2, 2, sharex=True, sharey=True,figsize=(8,6))

axs[0,0].hist(np.log(amyg_taus_m))
axs[0,0].set_title('amygdala')
axs[0,0].text(-10,175,'n_units = %i' %len(amyg_taus_m))

axs[0,1].hist(np.log(dacc_taus))
axs[0,1].set_title('dACC')
axs[0,1].text(-10,175,'n_units = %i' %len(dacc_taus))

axs[1,0].hist(np.log(hc_taus_m))
axs[1,0].set_title('hippocampus')
axs[1,0].text(-10,150,'n_units = %i' %len(hc_taus_m))

axs[1,1].hist(np.log(presma_taus))
axs[1,1].set_title('preSMA')
axs[1,1].text(-10,150,'n_units = %i' %len(presma_taus))

fig.add_subplot(111,frame_on=False)
plt.tick_params(labelcolor="none", bottom=False, left=False)

plt.xlabel("log(tau)")
plt.ylabel("count")
plt.title('Minxha human single units')

plt.show()