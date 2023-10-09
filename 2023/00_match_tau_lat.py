#%%

# imports

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

new_data = pd.read_csv('2023_timescales.csv')
old_data = pd.read_csv('old_data.csv')

#%%

matches = []

for unit in range(len(new_data)):
    
    this_unit = new_data.iloc[unit]
    
    tau = this_unit.tau
    lat = this_unit.lat
    r2 = this_unit.r2
    nspk = this_unit.nbSpk
    fr = this_unit.FR
    
    match_unit = old_data[(old_data.tau == tau) & (old_data.lat == lat)]
    
    if len(match_unit) > 0:
        
        dataset = match_unit.name
        area = match_unit.area
        ID = match_unit.unitID

        r2_ = match_unit.r2
        nspk_ = match_unit.nbSpk
        fr_ = match_unit.FR
        
        matches.append((dataset,area,ID,tau,lat,r2_,nspk_,fr_,r2,nspk,fr))
        
matches = pd.DataFrame(matches,columns=['dataset','area','unitID','tau','lat','old_r2','old_nspk','old_fr','new_r2','new_nspk','new_fr'])

#%%