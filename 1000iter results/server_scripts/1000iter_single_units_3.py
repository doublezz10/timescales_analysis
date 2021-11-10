# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 15:07:30 2021

@author: zachz
"""

import pandas as pd

from single_neuron_timescales import get_single_unit_timescales

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
        ('/home/zach/Wirth - monkey hippocampus/wirth_hippocampus2.mat','wirth','monkey','hc2')
        ]

all_data = pd.DataFrame()

for file in range(len(files)):
    
    for iteration in range(1000):
    
        print('Iteration #', iteration)

        df = get_single_unit_timescales(files[file][0],files[file][1],files[file][2],files[file][3])
        
        df['iter'] = iteration
        
        all_data = pd.concat([all_data,df],ignore_index=True)
    
all_data.to_csv('/home/zach/single_units3.csv')