#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 15:49:22 2020

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
axs[0,0].set_title('amygdala - Minxha')
axs[0,0].text(600,.07,'tau = %i' %pars_amy_m[1])

axs[0,1].errorbar(x_m,mean_hc_m,yerr=se_hc_m,label='original data')
axs[0,1].plot(x_m,func(x_m,*pars_hc_m),label='fit curve')
# axs[0,1].legend(loc='upper right')
axs[0,1].set_title('hippocampus - Minxha')
axs[0,1].text(600,.07,'tau = %i' %pars_hc_m[1])

axs[1,0].errorbar(x_m,mean_amygdala_f,yerr=se_amygdala_f,label='original data')
axs[1,0].plot(x_m,func(x_m,*pars_amy_f),label='fit curve')
# axs[1,0].legend(loc='upper right')
axs[1,0].set_title('amygdala - Faraut')
axs[1,0].text(600,.07,'tau = %i' %pars_amy_f[1])

axs[1,1].errorbar(x_m,mean_hc_f,yerr=se_hc_f,label='original data')
axs[1,1].plot(x_m,func(x_m,*pars_hc_f),label='fit curve')
# axs[1,1].legend(loc='upper right')
axs[1,1].set_title('hippocampus - Faraut')
axs[1,1].text(600,.07,'tau = %i' %pars_hc_f[1])

fig.add_subplot(111,frame_on=False)
plt.tick_params(labelcolor="none", bottom=False, left=False)

plt.xlabel("lag (ms)")
plt.ylabel("autocorrelation")

plt.show()

#%% Subplots of tau histograms

fig, axs = plt.subplots(2, 2, sharex=True, sharey=True,figsize=(8,6))

axs[0,0].hist(np.log(amyg_taus_m))
axs[0,0].set_title('amygdala - Minxha')
axs[0,0].text(-10,150,'n_units %i' %len(amyg_taus_m))

axs[0,1].hist(np.log(hc_taus_m))
axs[0,1].set_title('hippocampus - Minxha')
axs[0,1].text(-10,150,'n_units %i' %len(hc_taus_m))

axs[1,0].hist(np.log(amyg_taus_f))
axs[1,0].set_title('amygdala - Faraut')
axs[1,0].text(-10,150,'n_units %i' %len(amyg_taus_f))

axs[1,1].hist(np.log(hc_taus_f))
axs[1,1].set_title('hippocampus - Faraut')
axs[1,1].text(-10,150,'n_units %i' %len(hc_taus_f))

fig.add_subplot(111,frame_on=False)

plt.tick_params(labelcolor="none", bottom=False, left=False)

plt.xlabel("log(tau)")
plt.ylabel("count")

plt.show()