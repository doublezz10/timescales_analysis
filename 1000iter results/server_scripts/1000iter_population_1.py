# -*- coding: utf-8 -*-
"""
Created on Fri Mar 12 12:45:09 2021

@author: zachz
"""

from population_timescales import get_population_timescales
import pandas as pd

files = [('/home/zach/Buzsaki Rat/buzsaki_acc.mat','buzsaki','rat','acc'),
        ('/home/zach/Buzsaki Rat/buzsaki_mpfc.mat','buzsaki','rat','mpfc'),
        ('/home/zach/Buzsaki Rat/buzsaki_ofc.mat','buzsaki','rat','ofc'),
        ('/home/zach/Buzsaki2/buzsaki_bla.mat','buzsaki','rat','bla'),
        ('/home/zach/Buzsaki2/buzsaki_central.mat','buzsaki','rat','central'),
        ('/home/zach/Buzsaki2/buzsaki_hippocampus.mat','buzsaki','rat','hc'),
        ('/home/zach/LeMerre - rat mPFC/lemerre.mat','lemerre','rat','mpfc')
        ]

all_data = []

for file in range(len(files)):
    
    for iteration in range(1000):
        
        print('Iteration #', iteration)
    
        df = get_population_timescales(files[file][0],files[file][1],files[file][2],files[file][3])
        
        df = df + (iteration,)
        
        all_data.append(df)
        
all_data = pd.DataFrame(all_data,columns=['dataset','species','brain_area','tau','fr','n_units','prop_units','iter'])  
    
all_data.to_csv('/home/zach/pop_values1.csv')