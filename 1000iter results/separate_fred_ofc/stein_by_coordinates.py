#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.io import api

import seaborn as sns

from scipy.stats import linregress

from scipy.io import loadmat

plt.style.use('seaborn')

#%% Load in data, filter

raw_data = pd.read_csv('/Users/zachz/Documents/timescales_analysis/1000iter results/fixed_single_unit.csv')

data = raw_data[(raw_data.tau >= 10) & (raw_data.tau <= 1000)]

data = data[(data.r2 >= 0.5)]

data = data[data.species != 'rat']

listofspecies = ['mouse','monkey','human']

data['species'] = pd.Categorical(data['species'], categories = listofspecies , ordered = True)

_data = data[data.dataset=='steinmetz']

orb = _data[_data.brain_area=='orb']
bla = _data[_data.brain_area=='bla']

#%%

orb_cell_info = loadmat('/Users/zachz/Dropbox/Timescales across species/Spiketimes only/Steinmetz - mouse/unit_info/orb_unit_info.mat',simplify_cells=True)
bla_cell_info = loadmat('/Users/zachz/Dropbox/Timescales across species/Spiketimes only/Steinmetz - mouse/unit_info/bla_unit_info.mat',simplify_cells=True)

orb_cell_info = orb_cell_info['orb_units']
bla_cell_info = bla_cell_info['bla_units']


#%%

to_append = []
to_append2 = []

for unit in range(len(orb.unit.unique())):
    
    dataset = 'steinmetz'
    
    species = 'mouse'
    
    brain_area = 'orb'
    
    this_unit = orb[orb.unit == unit]
    
    mean_tau = np.mean(this_unit.tau)
    
    tau_sd = np.std(this_unit.tau)
    
    fr = np.mean(this_unit.fr)
    
    n = len(this_unit)
    
    r2 = np.mean(this_unit.r2)
    
    ap = orb_cell_info[unit]['Coordinates'][0]
    
    dv = orb_cell_info[unit]['Coordinates'][1]
    
    ml = orb_cell_info[unit]['Coordinates'][2]
    
    to_append.append((dataset,species,brain_area,unit + 1,mean_tau,tau_sd,n,r2,ap,dv,ml))
    
for unit in range(len(bla.unit.unique())):
    
    dataset = 'steinmetz'
    
    species = 'mouse'
    
    brain_area = 'bla'
    
    this_unit = orb[orb.unit == unit]
    
    mean_tau = np.mean(this_unit.tau)
    
    tau_sd = np.std(this_unit.tau)
    
    fr = np.mean(this_unit.fr)
    
    n = len(this_unit)
    
    r2 = np.mean(this_unit.r2)
    
    ap = bla_cell_info[unit]['Coordinates'][0]
    
    dv = bla_cell_info[unit]['Coordinates'][1]
    
    ml = bla_cell_info[unit]['Coordinates'][2]
    
    to_append2.append((dataset,species,brain_area,unit + 1,mean_tau,tau_sd,n,r2,ap,dv,ml))
    
orb_locations = pd.DataFrame(to_append,columns=['dataset','species','brain_area','unit','tau','tau_sd','n','r2','ap','dv','ml'])
bla_locations = pd.DataFrame(to_append2,columns=['dataset','species','brain_area','unit','tau','tau_sd','n','r2','ap','dv','ml'])

#%%

sns.scatterplot(data=orb_locations,x='tau',y='ap',size=0.3,alpha=0.6)

plt.legend(bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0.)

plt.xlabel('iterative tau (ms)')
plt.ylabel('a/p coordinate')

plt.title('Steinmetz ORB')

plt.show()

#%%

sns.scatterplot(data=orb_locations,x='tau',y='dv',size=0.3,alpha=0.6)

plt.legend(bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0.)

plt.xlabel('iterative tau (ms)')
plt.ylabel('d/v coordinate')

plt.title('Steinmetz ORB')

plt.show()

#%%

sns.scatterplot(data=orb_locations,x='tau',y='ml',size=0.3,alpha=0.6)

plt.legend(bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0.)

plt.xlabel('iterative tau (ms)')
plt.ylabel('m/l coordinate')

plt.title('Steinmetz ORB')

plt.show()


# %% try 3d plot

from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.add_subplot(111, projection = '3d')

x = orb_locations['ap']
y = orb_locations['ml']
z = orb_locations['dv']


taus = orb_locations['tau']

ax.scatter(x, y, z, c=taus,cmap='Blues')

ax.set_xlabel('A/P')
ax.set_zlabel('D/V')
ax.set_ylabel('M/L')


plt.show()

#%% Repeat with BLA

sns.scatterplot(data=bla_locations,x='tau',y='ap',size=0.3,alpha=0.6)

plt.legend(bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0.)

plt.xlabel('iterative tau (ms)')
plt.ylabel('a/p coordinate')

plt.title('Steinmetz BLA')

plt.show()

#%%

sns.scatterplot(data=bla_locations,x='tau',y='dv',size=0.3,alpha=0.6)

plt.legend(bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0.)

plt.xlabel('iterative tau (ms)')
plt.ylabel('d/v coordinate')

plt.title('Steinmetz BLA')

plt.show()

#%%

sns.scatterplot(data=bla_locations,x='tau',y='ml',size=0.3,alpha=0.6)

plt.legend(bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0.)

plt.xlabel('iterative tau (ms)')
plt.ylabel('m/l coordinate')

plt.title('Steinmetz BLA')

plt.show()


# %% try 3d plot

from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.add_subplot(111, projection = '3d')

x = bla_locations['ap']
y = bla_locations['ml']
z = bla_locations['dv']


taus = bla_locations['tau']

ax.scatter(x, y, z, c=taus,cmap='Blues')

ax.set_xlabel('A/P')
ax.set_zlabel('D/V')
ax.set_ylabel('M/L')


plt.show()
