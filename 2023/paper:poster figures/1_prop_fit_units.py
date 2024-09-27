#%%

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('seaborn')

data = pd.read_csv('processed_data.csv')

data['species'] = pd.Categorical(data['species'], categories = ['mouse','monkey','human'] , ordered = True)


filt_data = data[np.logical_and(data.tau < 1000, data.r2 > 0.5)]
filt_data = filt_data[filt_data.tau > 10]

# brain_regions = ['Hippocampus','OFC','Amygdala','mPFC','ACC']

# filt_data['brain_region'] = pd.Categorical(filt_data['brain_region'], categories = brain_regions , ordered = True)

#%%

# brain region data

plt.figure(figsize=(2.75,2.75))

f = sns.countplot(data=data,x='brain_region',hue='species',alpha=0.2,order=['Hippocampus','OFC','Amygdala','mPFC','ACC'])
g = sns.countplot(data=filt_data,x='brain_region',hue='species',order=['Hippocampus','OFC','Amygdala','mPFC','ACC'])

handles, labels = g.get_legend_handles_labels()
plt.legend(handles[3:], labels[3:],loc='upper left',prop={'size':7})
plt.xlabel('')
plt.ylabel('number of neurons',fontsize=7)
plt.xticks(fontsize=7,rotation=45)
plt.yticks(fontsize=7)

plt.show()

#%%

from scipy.stats import sem

# Define the decay function
def decay_function(z, tau, A, B):
    return A * (np.exp(-z / tau) + B)

# Group the data by species and brain region
grouped = filt_data.groupby(['species', 'brain_region'])

# Generate a range of z values
z_values = np.linspace(0, 800) # Adjust the range and number of points as needed

# Get unique brain regions

brain_regions = ['Hippocampus','Amygdala','ACC']


# Create subplots
fig, axes = plt.subplots(1, len(brain_regions), sharex=True,figsize=(2.75*len(brain_regions),2.75))

# Iterate over each brain region
for ax, brain_region in zip(axes, brain_regions):
    # Filter data for the current brain region
    region_data = filt_data[filt_data['brain_region'] == brain_region]
    
    # Group by species and calculate mean and SEM
    species_grouped = region_data.groupby('species').agg(['mean', sem])
    
    # Iterate over each species
    for species, params in species_grouped.iterrows():
        tau_mean = params[('tau', 'mean')]
        A_mean = params[('A', 'mean')]
        B_mean = params[('B', 'mean')]
        
        tau_sem = params[('tau', 'sem')]
        A_sem = params[('A', 'sem')]
        B_sem = params[('B', 'sem')]
        # Compute the decay curve
        decay_curve = decay_function(z_values, tau_mean, A_mean, B_mean)
        
        # Compute the confidence interval
        decay_curve_upper = decay_function(z_values, tau_mean + tau_sem, A_mean + A_sem, B_mean + B_sem)
        decay_curve_lower = decay_function(z_values, tau_mean - tau_sem, A_mean - A_sem, B_mean - B_sem)
        
        # Plot the decay curve
        ax.plot(z_values, decay_curve, label=f'{species} (mean $\\tau$={tau_mean:.0f} ms)')
        ax.fill_between(z_values, decay_curve_lower, decay_curve_upper, alpha=0.3)
    
    ax.set_title(f'{brain_region}',fontsize=7)
    ax.set_xlabel('time lag (ms)',fontsize=7)
    # make tick labels font size 7
    ax.tick_params(axis='both',labelsize=7)
    ax.legend(fontsize=7)

    ax.set_ylabel('')
     
    if brain_region == 'Hippocampus':
        ax.set_ylim(0,7)
        ax.set_ylabel('autocorrelation (a.u.)',fontsize=7)

    elif brain_region == 'Amygdala':
        ax.set_ylim(0,7)

plt.tight_layout()
plt.show()

# %%
