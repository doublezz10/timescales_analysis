#%%

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns

import matplotlib

plt.style.use('seaborn')

plt.rcParams['font.size'] = '7'

listofspecies = ['mouse','monkey','human']

fred_data = pd.read_csv('/Users/zachz/Documents/timescales_analysis/1000iter results/isi figs/filtered_isi_data.csv')

fred_data['species'] = pd.Categorical(fred_data['species'], categories=listofspecies, ordered=True)

fred_brain_region_data2 = fred_data

#%%
all_ns=[]

for species in fred_brain_region_data2.species.unique():
    
    this_species = fred_brain_region_data2[fred_brain_region_data2.species==species]
    
    for brain_area in this_species.brain_region.unique():
        
        n_neurons = len(this_species[this_species.brain_region == brain_area])

        all_ns.append((species,brain_area,n_neurons))
        
all_ns = pd.DataFrame(all_ns,columns=['species','brain_region','n_neurons'])

listofspecies = ['mouse','monkey','human']

all_ns['species'] = pd.Categorical(all_ns['species'], categories = listofspecies , ordered = True)

total_ns = [298,576+229,189,230,2843,329,1086+460,759+203,320,841,317,2800,1274,813]
# change this to only those that meet fr criteria

all_ns['total_n'] = np.array(total_ns)

#%%
plt.figure(figsize=(1.75,3.2))

sns.scatterplot(x='brain_region',y='n_neurons',hue='species',style='species',data=all_ns)

plt.tick_params(axis='x', rotation=90,labelsize=7)
plt.tick_params(axis='y',labelsize=7)

plt.legend(loc='upper left',prop={'size':7})

plt.xlabel(None)
plt.ylabel('number of neurons',fontdict={'fontsize':7})

plt.tight_layout()

plt.show()

#%%
plt.figure(figsize=(2.4,3.2))

sns.barplot(x='brain_region',y='n_neurons',hue='species',data=all_ns,order=['Hippocampus','OFC','Amygdala','mPFC','ACC'])

plt.tick_params(axis='x', rotation=90,labelsize=7)
plt.tick_params(axis='y',labelsize=7)

ax = sns.barplot(x='brain_region',y='total_n',hue='species',data=all_ns,alpha=0.3,order=['Hippocampus','OFC','Amygdala','mPFC','ACC'])

plt.xlabel(None)

handles, labels = ax.get_legend_handles_labels()
plt.legend(handles[0:3], labels[0:3],loc='upper right',prop={'size':7})

plt.ylabel('number of neurons',fontdict={'fontsize':7})

plt.tight_layout()

plt.show()

#%%