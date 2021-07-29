#%%

"""
Created on Tue Jun 15 14:59:59 2021

@author: zachz
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm
import ptitprince as pt

import seaborn as sns

plt.style.use('seaborn')

#%% Load in data, filter

raw_data = pd.read_csv('/Users/zachz/Documents/timescales_analysis/1000iter results/fixed_single_unit.csv')

listofspecies = ['mouse','rat','monkey','human']

raw_data['species'] = pd.Categorical(raw_data['species'], categories = listofspecies , ordered = True)

data = raw_data[(raw_data.tau >= 10) & (raw_data.tau <= 1000)]

data = data[(data.r2 >= 0.5)]

data2 = data[(data.r2 >= 0.85)]

#%% Separate by brain_region

# raw data

raw_acc = raw_data[(raw_data.brain_area == 'acc') | (raw_data.brain_area == 'dACC') | (raw_data.brain_area == 'aca') | (raw_data.brain_area == 'mcc')]

raw_amyg = raw_data[(raw_data.brain_area == 'amygdala') | (raw_data.brain_area == 'central') | (raw_data.brain_area == 'bla')]

raw_hc = raw_data[(raw_data.brain_area == 'hc') | (raw_data.brain_area == 'hc2') | (raw_data.brain_area == 'ca1') | (raw_data.brain_area == 'ca2') | (raw_data.brain_area == 'ca3') | (raw_data.brain_area == 'dg')]

raw_mpfc = raw_data[(raw_data.brain_area == 'mpfc') | (raw_data.brain_area == 'pl') | (raw_data.brain_area == 'ila') | (raw_data.brain_area == 'scACC')]

raw_ofc = raw_data[(raw_data.brain_area == 'ofc') | (raw_data.brain_area == 'orb')]
           
raw_striatum = raw_data[(raw_data.brain_area == 'vStriatum') | (raw_data.brain_area == 'putamen') | (raw_data.brain_area == 'caudate')]

# filtered data at both thresholds

acc = data[(data.brain_area == 'acc') | (data.brain_area == 'dACC') | (data.brain_area == 'aca') | (data.brain_area == 'mcc')]

amyg = data[(data.brain_area == 'amygdala') | (data.brain_area == 'central') | (data.brain_area == 'bla')]

hc = data[(data.brain_area == 'hc') | (data.brain_area == 'ca1') | (data.brain_area == 'ca2') | (data.brain_area == 'ca3') | (data.brain_area == 'dg')]

mpfc = data[(data.brain_area == 'mpfc') | (data.brain_area == 'pl') | (data.brain_area == 'ila') | (data.brain_area == 'scACC')]

ofc = data[(data.brain_area == 'ofc') | (data.brain_area == 'orb')]
           
striatum = data[(data.brain_area == 'vStriatum') | (data.brain_area == 'putamen') | (data.brain_area == 'caudate')]

brain_regions = pd.concat((acc,amyg,hc,mpfc,ofc,striatum))

acc2 = data2[(data2.brain_area == 'acc') | (data2.brain_area == 'dACC') | (data2.brain_area == 'aca') | (data2.brain_area == 'mcc')]

amyg2 = data2[(data2.brain_area == 'amygdala') | (data2.brain_area == 'central') | (data2.brain_area == 'bla')]

hc2 = data2[(data2.brain_area == 'hc') | (data2.brain_area == 'ca1') | (data2.brain_area == 'ca2') | (data2.brain_area == 'ca3') | (data2.brain_area == 'dg')]

mpfc2 = data2[(data2.brain_area == 'mpfc') | (data2.brain_area == 'pl') | (data2.brain_area == 'ila') | (data2.brain_area == 'scACC')]

ofc2 = data2[(data2.brain_area == 'ofc') | (data2.brain_area == 'orb')]
           
striatum2 = data2[(data2.brain_area == 'vStriatum') | (data2.brain_area == 'putamen') | (data2.brain_area == 'caudate')]

brain_regions2 = pd.concat((acc,amyg,hc,mpfc,ofc,striatum))

#%% filter for rats only

raw_rat_acc = raw_acc[raw_acc.species=='rat']
raw_rat_amyg = raw_amyg[raw_amyg.species=='rat']
raw_rat_hc = raw_hc[raw_hc.species=='rat']
raw_rat_mpfc = raw_mpfc[raw_mpfc.species=='rat']
raw_rat_ofc = raw_ofc[raw_ofc.species=='rat']
raw_rat_striatum = raw_striatum[raw_striatum.species=='rat']

rat_acc = acc[acc.species=='rat']
rat_amyg = amyg[amyg.species=='rat']
rat_hc = hc[hc.species=='rat']
rat_mpfc = mpfc[mpfc.species=='rat']
rat_ofc = ofc[ofc.species=='rat']
rat_striatum = striatum[striatum.species=='rat']

rat_acc2 = acc2[acc2.species=='rat']
rat_amyg2 = amyg2[amyg2.species=='rat']
rat_hc2 = hc2[hc2.species=='rat']
rat_mpfc2 = mpfc2[mpfc2.species=='rat']
rat_ofc2 = ofc2[ofc2.species=='rat']
rat_striatum2 = striatum2[striatum2.species=='rat']

#%% how many units don't have very many fits per dataset

# ACC

bins = np.linspace(0,1,21)

for dataset in rat_acc.dataset.unique():
    
    this_dataset = rat_acc[rat_acc.dataset == dataset]
    
    other_dataset = rat_acc2[rat_acc2.dataset == dataset]
    
    fig,axs = plt.subplots(1,2,sharey = True)
    
    dist_5 = []
    dist_85 = []
    
    for unit_n in this_dataset.unit.unique():
        
        this_unit = this_dataset[this_dataset.unit == unit_n]
        
        r2s = this_unit['r2']
        
        r2_dist = np.histogram(r2s,bins=bins)[0]
        
        dist_5.append(r2_dist)
        
        try:
            
            other_unit = other_dataset[other_dataset.unit == unit_n]
            
            other_r2s = other_unit['r2']
            
            other_dist = np.histogram(other_r2s,bins=bins)[0]
            
            dist_85.append(other_dist)
            
        except:
            
            other_r2s = []
            
            dist_85.append([])

            pass
    
    dist_5 = np.vstack(dist_5)
    dist_85 = np.vstack(dist_85)
    
    bins = np.around(bins[:-1],decimals=3)
    
    sns.heatmap(dist_5,yticklabels=False,xticklabels=bins,cbar_kws={"label": "n iterations",'orientation': 'horizontal'},ax=axs[0],vmin=0,vmax=1000)
    sns.heatmap(dist_85,yticklabels=False,xticklabels=bins,cbar_kws={"label": "n iterations",'orientation': 'horizontal'},ax=axs[1],vmin=0,vmax=1000)

    axs[0].set_title('0.5 R$^2$ threshold')
    axs[1].set_title('0.85 R$^2$ threshold')
    
    axs[0].set_ylabel('unit')
    axs[0].set_xlabel('R$^2$')
    axs[1].set_xlabel('R$^2$')

    plt.suptitle(dataset + ' Rat ACC')
        
    plt.show()
    
# amyg

bins = np.linspace(0,1,21)

for dataset in rat_amyg.dataset.unique():
    
    this_dataset = rat_amyg[rat_amyg.dataset == dataset]
    
    other_dataset = rat_amyg2[rat_amyg2.dataset == dataset]
    
    fig,axs = plt.subplots(1,2,sharey = True)
    
    dist_5 = []
    dist_85 = []
    
    for unit_n in this_dataset.unit.unique():
        
        this_unit = this_dataset[this_dataset.unit == unit_n]
        
        r2s = this_unit['r2']
        
        r2_dist = np.histogram(r2s,bins=bins)[0]
        
        dist_5.append(r2_dist)
        
        try:
            
            other_unit = other_dataset[other_dataset.unit == unit_n]
            
            other_r2s = other_unit['r2']
            
            other_dist = np.histogram(other_r2s,bins=bins)[0]
            
            dist_85.append(other_dist)
            
        except:
            
            other_r2s = []
            
            dist_85.append([])

            pass
    
    dist_5 = np.vstack(dist_5)
    dist_85 = np.vstack(dist_85)
    
    bins = np.around(bins[:-1],decimals=3)
    
    sns.heatmap(dist_5,yticklabels=False,xticklabels=bins,cbar_kws={"label": "n iterations",'orientation': 'horizontal'},ax=axs[0],vmin=0,vmax=1000)
    sns.heatmap(dist_85,yticklabels=False,xticklabels=bins,cbar_kws={"label": "n iterations",'orientation': 'horizontal'},ax=axs[1],vmin=0,vmax=1000)

    axs[0].set_title('0.5 R$^2$ threshold')
    axs[1].set_title('0.85 R$^2$ threshold')
    
    axs[0].set_ylabel('unit')
    axs[0].set_xlabel('R$^2$')
    axs[1].set_xlabel('R$^2$')
    
    plt.suptitle(dataset + ' Rat Amygdala')
        
    plt.show()
    
# hc

bins = np.linspace(0,1,21)

for dataset in rat_hc.dataset.unique():
    
    this_dataset = rat_hc[rat_hc.dataset == dataset]
    
    other_dataset = rat_hc2[rat_hc2.dataset == dataset]
    
    fig,axs = plt.subplots(1,2,sharey = True)
    
    dist_5 = []
    dist_85 = []
    
    for unit_n in this_dataset.unit.unique():
        
        this_unit = this_dataset[this_dataset.unit == unit_n]
        
        r2s = this_unit['r2']
        
        r2_dist = np.histogram(r2s,bins=bins)[0]
        
        dist_5.append(r2_dist)
        
        try:
            
            other_unit = other_dataset[other_dataset.unit == unit_n]
            
            other_r2s = other_unit['r2']
            
            other_dist = np.histogram(other_r2s,bins=bins)[0]
            
            dist_85.append(other_dist)
            
        except:
            
            other_r2s = []
            
            dist_85.append([])

            pass
    
    dist_5 = np.vstack(dist_5)
    dist_85 = np.vstack(dist_85)
    
    bins = np.around(bins[:-1],decimals=3)
    
    sns.heatmap(dist_5,yticklabels=False,xticklabels=bins,cbar_kws={"label": "n iterations",'orientation': 'horizontal'},ax=axs[0],vmin=0,vmax=1000)
    sns.heatmap(dist_85,yticklabels=False,xticklabels=bins,cbar_kws={"label": "n iterations",'orientation': 'horizontal'},ax=axs[1],vmin=0,vmax=1000)

    axs[0].set_title('0.5 R$^2$ threshold')
    axs[1].set_title('0.85 R$^2$ threshold')
    
    axs[0].set_ylabel('unit')
    axs[0].set_xlabel('R$^2$')
    axs[1].set_xlabel('R$^2$')

    plt.suptitle(dataset + ' Rat Hippocampus')
        
    plt.show()
    
# mPFC

bins = np.linspace(0,1,21)

for dataset in rat_mpfc.dataset.unique():
    
    this_dataset = rat_mpfc[rat_mpfc.dataset == dataset]
    
    other_dataset = rat_mpfc2[rat_mpfc2.dataset == dataset]
    
    fig,axs = plt.subplots(1,2,sharey = True)
    
    dist_5 = []
    dist_85 = []
    
    for unit_n in this_dataset.unit.unique():
        
        this_unit = this_dataset[this_dataset.unit == unit_n]
        
        r2s = this_unit['r2']
        
        r2_dist = np.histogram(r2s,bins=bins)[0]
        
        dist_5.append(r2_dist)
        
        try:
            
            other_unit = other_dataset[other_dataset.unit == unit_n]
            
            other_r2s = other_unit['r2']
            
            other_dist = np.histogram(other_r2s,bins=bins)[0]
            
            dist_85.append(other_dist)
            
        except:
            
            other_r2s = []
            
            dist_85.append([])

            pass
    
    dist_5 = np.vstack(dist_5)
    dist_85 = np.vstack(dist_85)
    
    bins = np.around(bins[:-1],decimals=3)
    
    sns.heatmap(dist_5,yticklabels=False,xticklabels=bins,cbar_kws={"label": "n iterations",'orientation': 'horizontal'},ax=axs[0],vmin=0,vmax=1000)
    sns.heatmap(dist_85,yticklabels=False,xticklabels=bins,cbar_kws={"label": "n iterations",'orientation': 'horizontal'},ax=axs[1],vmin=0,vmax=1000)

    axs[0].set_title('0.5 R$^2$ threshold')
    axs[1].set_title('0.85 R$^2$ threshold')
    
    axs[0].set_ylabel('unit')
    axs[0].set_xlabel('R$^2$')
    axs[1].set_xlabel('R$^2$')
    
    plt.suptitle(dataset + ' Rat mPFC')
        
    plt.show()
    
# OFC

bins = np.linspace(0,1,21)

for dataset in rat_ofc.dataset.unique():
    
    this_dataset = rat_ofc[rat_ofc.dataset == dataset]
    
    other_dataset = rat_ofc2[rat_ofc2.dataset == dataset]
    
    fig,axs = plt.subplots(1,2,sharey = True)
    
    dist_5 = []
    dist_85 = []
    
    for unit_n in this_dataset.unit.unique():
        
        this_unit = this_dataset[this_dataset.unit == unit_n]
        
        r2s = this_unit['r2']
        
        r2_dist = np.histogram(r2s,bins=bins)[0]
        
        dist_5.append(r2_dist)
        
        try:
            
            other_unit = other_dataset[other_dataset.unit == unit_n]
            
            other_r2s = other_unit['r2']
            
            other_dist = np.histogram(other_r2s,bins=bins)[0]
            
            dist_85.append(other_dist)
            
        except:
            
            other_r2s = []
            
            dist_85.append([])

            pass
    
    dist_5 = np.vstack(dist_5)
    dist_85 = np.vstack(dist_85)
    
    bins = np.around(bins[:-1],decimals=3)
    
    sns.heatmap(dist_5,yticklabels=False,xticklabels=bins,cbar_kws={"label": "n iterations",'orientation': 'horizontal'},ax=axs[0],vmin=0,vmax=1000)
    sns.heatmap(dist_85,yticklabels=False,xticklabels=bins,cbar_kws={"label": "n iterations",'orientation': 'horizontal'},ax=axs[1],vmin=0,vmax=1000)

    axs[0].set_title('0.5 R$^2$ threshold')
    axs[1].set_title('0.85 R$^2$ threshold')
    
    axs[0].set_ylabel('unit')
    axs[0].set_xlabel('R$^2$')
    axs[1].set_xlabel('R$^2$')
    
    plt.suptitle(dataset + ' Rat OFC')
        
    plt.show()



#%% Violin plots by area

# ACC

for dataset in rat_acc.dataset.unique():
    
    this_dataset = rat_acc[rat_acc.dataset == dataset]
    
    other_dataset = rat_acc2[rat_acc2.dataset == dataset]
    
    for unit_n in this_dataset.unit.unique():
        
        this_unit = this_dataset[this_dataset.unit == unit_n]
        
        taus = this_unit['tau']
        
        fig,axs = plt.subplots(1,2,sharey=True)
        
        sns.violinplot(ax=axs[0],y=taus)
        
        try:
        
            other_unit = other_dataset[other_dataset.unit == unit_n]
            
            taus2 = other_unit['tau']
        
            sns.violinplot(ax=axs[1],y=taus2)
            
        except:
            
            taus2 = []
            
            pass
        
        axs[0].set_title('R$^2$ > 0.5 \n %i iters' %len(taus))
        axs[1].set_title('R$^2$ > 0.85 \n %i iters' %len(taus2))
        
        axs[1].set_ylabel('')
        
        plt.suptitle(dataset + ' Rat ACC unit %i' %unit_n)
        
        plt.show()
        
# Amygdala
        
for dataset in rat_amyg.dataset.unique():
    
    this_dataset = rat_amyg[rat_amyg.dataset == dataset]
    
    other_dataset = rat_amyg2[rat_amyg2.dataset == dataset]
    
    for unit_n in this_dataset.unit.unique():
        
        this_unit = this_dataset[this_dataset.unit == unit_n]
        
        taus = this_unit['tau']
        
        fig,axs = plt.subplots(1,2,sharey=True)
        
        sns.violinplot(ax=axs[0],y=taus)
        
        try:
        
            other_unit = other_dataset[other_dataset.unit == unit_n]
            
            taus2 = other_unit['tau']
        
            sns.violinplot(ax=axs[1],y=taus2)
            
        except:
            
            taus2 = []
            
            pass
        
        axs[0].set_title('R$^2$ > 0.5 \n %i iters' %len(taus))
        axs[1].set_title('R$^2$ > 0.85 \n %i iters' %len(taus2))
        
        axs[1].set_ylabel('')
        
        plt.suptitle(dataset + ' Rat amygdala unit %i' %unit_n)
        
        plt.show()
        
# Hippocampus

for dataset in rat_hc.dataset.unique():
    
    this_dataset = rat_hc[rat_hc.dataset == dataset]
    
    other_dataset = rat_hc2[rat_hc2.dataset == dataset]
    
    for unit_n in this_dataset.unit.unique():
        
        this_unit = this_dataset[this_dataset.unit == unit_n]
        
        taus = this_unit['tau']
        
        fig,axs = plt.subplots(1,2,sharey=True)
        
        sns.violinplot(ax=axs[0],y=taus)
        
        try:
        
            other_unit = other_dataset[other_dataset.unit == unit_n]
            
            taus2 = other_unit['tau']
        
            sns.violinplot(ax=axs[1],y=taus2)
            
        except:
            
            taus2 = []
            
            pass
        
        axs[0].set_title('R$^2$ > 0.5 \n %i iters' %len(taus))
        axs[1].set_title('R$^2$ > 0.85 \n %i iters' %len(taus2))
        
        axs[1].set_ylabel('')

        plt.suptitle(dataset + ' Rat hippocampus unit %i' %unit_n)
        
        plt.show()
        
# mPFC

for dataset in rat_mpfc.dataset.unique():
    
    this_dataset = rat_mpfc[rat_mpfc.dataset == dataset]
    
    other_dataset = rat_mpfc2[rat_mpfc2.dataset == dataset]
    
    for unit_n in this_dataset.unit.unique():
        
        this_unit = this_dataset[this_dataset.unit == unit_n]
        
        taus = this_unit['tau']
        
        fig,axs = plt.subplots(1,2,sharey=True)
        
        sns.violinplot(ax=axs[0],y=taus)
        
        try:
        
            other_unit = other_dataset[other_dataset.unit == unit_n]
            
            taus2 = other_unit['tau']
        
            sns.violinplot(ax=axs[1],y=taus2)
            
        except:
            
            taus2 = []
            
            pass
        
        axs[0].set_title('R$^2$ > 0.5 \n %i iters' %len(taus))
        axs[1].set_title('R$^2$ > 0.85 \n %i iters' %len(taus2))
        
        axs[1].set_ylabel('')

        plt.suptitle(dataset + ' Rat mPFC unit %i' %unit_n)
        
        plt.show()
        
# OFC
        
for dataset in rat_ofc.dataset.unique():
    
    this_dataset = rat_ofc[rat_ofc.dataset == dataset]
    
    other_dataset = rat_ofc2[rat_ofc2.dataset == dataset]
    
    for unit_n in this_dataset.unit.unique():
        
        this_unit = this_dataset[this_dataset.unit == unit_n]
        
        taus = this_unit['tau']
        
        fig,axs = plt.subplots(1,2,sharey=True)
        
        sns.violinplot(ax=axs[0],y=taus)
        
        try:
        
            other_unit = other_dataset[other_dataset.unit == unit_n]
            
            taus2 = other_unit['tau']
        
            sns.violinplot(ax=axs[1],y=taus2)
            
        except:
            
            taus2 = []
            
            pass
        
        axs[0].set_title('R$^2$ > 0.5 \n %i iters' %len(taus))
        axs[1].set_title('R$^2$ > 0.85 \n %i iters' %len(taus2))
        
        axs[1].set_ylabel('')
        
        plt.suptitle(dataset + ' Rat OFC unit %i' %unit_n)
        
        plt.show()
        
# Striatum

for dataset in rat_striatum.dataset.unique():
    
    this_dataset = rat_striatum[rat_striatum.dataset == dataset]
    
    other_dataset = rat_striatum2[rat_striatum2.dataset == dataset]
    
    for unit_n in this_dataset.unit.unique():
        
        this_unit = this_dataset[this_dataset.unit == unit_n]
        
        taus = this_unit['tau']
        
        fig,axs = plt.subplots(1,2,sharey=True)
        
        sns.violinplot(ax=axs[0],y=taus)
        
        try:
        
            other_unit = other_dataset[other_dataset.unit == unit_n]
            
            taus2 = other_unit['tau']
        
            sns.violinplot(ax=axs[1],y=taus2)
            
        except:
            
            taus2 = []
            
            pass
        
        axs[0].set_title('R$^2$ > 0.5 \n %i iters' %len(taus))
        axs[1].set_title('R$^2$ > 0.85 \n %i iters' %len(taus2))
        
        axs[1].set_ylabel('')
        
        plt.suptitle(dataset + ' Rat striatum unit %i' %unit_n)
        
        plt.show()