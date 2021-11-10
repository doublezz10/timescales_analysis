# -*- coding: utf-8 -*-
"""
Created on Fri Mar 12 12:45:09 2021

@author: zachz
"""

from population_timescales import get_population_timescales
import pandas as pd


files = [('/home/zach/Steinmetz - mouse/stein_orb.mat','steinmetz','mouse','orb'),
        ('/home/zach/Steinmetz - mouse/stein_pl.mat','steinmetz','mouse','pl'),
        ('/home/zach/Stoll - monkey1/stoll_AMG.mat','stoll','monkey','amygdala'),
        ('/home/zach/Stoll - monkey1/stoll_Cd.mat','stoll','monkey','caudate'),
        ('/home/zach/Stoll - monkey1/stoll_dlPFC.mat','stoll','monkey','dlPFC'),
        ('/home/zach/Stoll - monkey1/stoll_IFG.mat','stoll','monkey','IFG'),
        ('/home/zach/Stoll - monkey1/stoll_LAI.mat','stoll','monkey','LAI'),
        ('/home/zach/Stoll - monkey1/stoll_OFC.mat','stoll','monkey','ofc'),
        ('/home/zach/Stoll - monkey1/stoll_PMd.mat','stoll','monkey','PMd'),
        ('/home/zach/Stoll - monkey1/stoll_PUT.mat','stoll','monkey','putamen'),
        ('/home/zach/Stoll - monkey1/stoll_vlPFC.mat','stoll','monkey','vlPFC'),
        ('/home/zach/Wirth - monkey hippocampus/wirth_hippocampus.mat','wirth','monkey','hc'),
        ('/home/zach/Wirth - monkey hippocampus/wirth_hippocampus2.mat','wirth','monkey','hc')
        ]

all_data = []

for file in range(len(files)):
    
    for iteration in range(1000):
        
        print('Iteration #', iteration)
    
        df = get_population_timescales(files[file][0],files[file][1],files[file][2],files[file][3])
        
        df = df + (iteration,)
        
        all_data.append(df)
        
all_data = pd.DataFrame(all_data,columns=['dataset','species','brain_area','tau','fr','n_units','prop_units','iter'])  
    
all_data.to_csv('/home/zach/pop_values3.csv')