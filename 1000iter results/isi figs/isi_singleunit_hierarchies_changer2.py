#%%

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns

import matplotlib

plt.style.use('seaborn')

plt.rcParams['font.size'] = '7'

listofspecies = ['mouse','monkey','human']

for r2 in [0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95]:

    fred_data = pd.read_csv('/Users/zachz/Documents/timescales_analysis/1000iter results/fred_data.csv')
    fred_data = fred_data.rename(columns={'unitID': 'unit', 'name': 'dataset', 'area': 'brain_area'})
    fred_data = fred_data[fred_data.dataset != 'faraut']

    fred_data['species'] = pd.Categorical(fred_data['species'], categories=listofspecies, ordered=True)

    # rename columns to match

    fred_data['brain_area'] = fred_data['brain_area'].str.replace('hippocampus','hc')
    fred_data['brain_area'] = fred_data['brain_area'].str.replace('mPFC','mpfc')
    fred_data['brain_area'] = fred_data['brain_area'].str.replace('ventralStriatum','vStriatum')
    fred_data['brain_area'] = fred_data['brain_area'].str.replace('Cd','caudate')
    fred_data['brain_area'] = fred_data['brain_area'].str.replace('OFC','ofc')
    fred_data['brain_area'] = fred_data['brain_area'].str.replace('PUT','putamen')
    fred_data['brain_area'] = fred_data['brain_area'].str.replace('hippocampus2','hc2')

    fred_data = fred_data.replace(['amyg','AMG'],'amygdala')

    fred_data['dataset'] = fred_data['dataset'].str.replace('stein','steinmetz')

    fred_data = fred_data[fred_data.r2 >= r2]

    fred_data = fred_data[(fred_data.tau >=10) & (fred_data.tau <= 1000)]
    
    acc = fred_data[(fred_data.brain_area == 'acc') | (fred_data.brain_area == 'dACC') | (fred_data.brain_area == 'aca') | (fred_data.brain_area == 'mcc')]

    amyg = fred_data[(fred_data.brain_area == 'amygdala') | (fred_data.brain_area == 'central') | (fred_data.brain_area == 'bla')]

    hc = fred_data[(fred_data.brain_area == 'hc') | (fred_data.brain_area == 'hc2') | (fred_data.brain_area == 'ca1') | (fred_data.brain_area == 'ca2') | (fred_data.brain_area == 'ca3') | (fred_data.brain_area == 'dg')]

    mpfc = fred_data[(fred_data.brain_area == 'mpfc') | (fred_data.brain_area == 'pl') | (fred_data.brain_area == 'ila') | (fred_data.brain_area == 'scACC')]

    ofc = fred_data[(fred_data.brain_area == 'ofc') | (fred_data.brain_area == 'orb')]

    lai = fred_data[fred_data.brain_area == 'LAI']

    vlpfc = fred_data[fred_data.brain_area == 'vlPFC']


    acc2 = acc.assign(brain_region='ACC')
    amyg2 = amyg.assign(brain_region='Amygdala')
    hc2 = hc.assign(brain_region='Hippocampus')
    mpfc2 = mpfc.assign(brain_region='mPFC')
    ofc2 = ofc.assign(brain_region='OFC')
    lai2 = lai.assign(brain_region='LAI')
    vlpfc2 = vlpfc.assign(brain_region='vlPFC')

    fred_brain_region_data2 = pd.concat((acc2,amyg2,hc2,mpfc2,ofc2))

#   First fig (no LAI)

    brain_regions2 = ['Hippocampus','Amygdala','OFC','mPFC','ACC']

    fred_brain_region_data2['brain_region'] = pd.Categorical(fred_brain_region_data2['brain_region'], categories = brain_regions2 , ordered = True)

    fig, axs = plt.subplots(1,2,figsize=(4.475,2.75))

    sns.lineplot(ax=axs[0],data=fred_brain_region_data2[fred_brain_region_data2.brain_region != 'LAI'],x='brain_region',y='tau',hue='species',ci=95,markers=True,legend=True,estimator=np.mean)

    axs[0].set_xlabel(None)
    axs[0].tick_params(axis='x', rotation=90,labelsize=7)
    axs[0].tick_params(axis='y',labelsize=7)
    axs[0].set_ylabel('timescale (ms)',fontsize=7)

    axs[0].legend(title='',prop={'size': 7})

    sns.lineplot(ax=axs[1],data=fred_brain_region_data2[fred_brain_region_data2.brain_region != 'LAI'],x='brain_region',y='lat',hue='species',ci=95,markers=True,legend=False,estimator=np.mean)

    axs[1].set_xlabel(None)
    axs[1].tick_params(axis='x', rotation=90,labelsize=7)
    axs[1].tick_params(axis='y',labelsize=7)
    axs[1].set_ylabel('latency (ms)',fontsize=7)
    
    plt.suptitle('R$^2$ thresh = %.2f' %(r2))

    plt.tight_layout()

    plt.show()

#%%