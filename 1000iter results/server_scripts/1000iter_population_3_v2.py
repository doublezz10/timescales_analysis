# -*- coding: utf-8 -*-
"""
Created on Fri Mar 12 12:45:09 2021

@author: zachz
"""

from population_timescales import get_population_timescales
import pandas as pd


files = [
        ('/home/zach/Wirth - monkey hippocampus/wirth_hippocampus.mat','wirth','monkey','hc'),
        ('/home/zach/Wirth - monkey hippocampus/wirth_hippocampus2.mat','wirth','monkey','hc2')
        ]

all_data = []

for file in range(len(files)):
    
    for iteration in range(1000):
        
        print('Iteration #', iteration)
    
        df = get_population_timescales(files[file][0],files[file][1],files[file][2],files[file][3])
        
        df = df + (iteration,)
        
        all_data.append(df)
        
all_data = pd.DataFrame(all_data,columns=['dataset','species','brain_area','tau','fr','n_units','prop_units','iter'])  
    
all_data.to_csv('/home/zach/pop_values3-2.csv')