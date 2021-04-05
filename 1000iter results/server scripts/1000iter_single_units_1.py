# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 15:05:35 2021

@author: zachz
"""

import pandas as pd

from single_neuron_timescales import get_single_unit_timescales

files = [('/home/zach/Buzsaki Rat/buzsaki_acc.mat','buzsaki','rat','acc'),
        ('/home/zach/Buzsaki Rat/buzsaki_mpfc.mat','buzsaki','rat','mpfc'),
        ('/home/zach/Buzsaki Rat/buzsaki_ofc.mat','buzsaki','rat','ofc'),
        ('/home/zach/Buzsaki2/buzsaki_bla.mat','buzsaki','rat','bla'),
        ('/home/zach/Buzsaki2/buzsaki_central.mat','buzsaki','rat','central'),
        ('/home/zach/Buzsaki2/buzsaki_hippocampus.mat','buzsaki','rat','hc'),
        ('/home/zach/LeMerre - rat mPFC/lemerre.mat','lemerre','rat','mpfc')
        ]

all_data = pd.DataFrame()

for file in range(len(files)):
    
    for iteration in range(1000):
    
        print('Iteration #', iteration)

        df = get_single_unit_timescales(files[file][0],files[file][1],files[file][2],files[file][3])
        
        df['iter'] = iteration
        
        all_data = pd.concat([all_data,df],ignore_index=True)
    
all_data.to_csv('/home/zach/single_units1.csv')