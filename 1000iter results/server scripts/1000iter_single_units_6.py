# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 15:07:30 2021

@author: zachz
"""

import pandas as pd

from single_neuron_timescales import get_single_unit_timescales

files = [
        ('/home/zach/Minxha - Human MFC/minxha_OFC.mat','minxha','human','ofc'),
        ]

all_data = pd.DataFrame()

for file in range(len(files)):
    
    for iteration in range(1000):
    
        print('Iteration #', iteration)

        df = get_single_unit_timescales(files[file][0],files[file][1],files[file][2],files[file][3])
        
        df['iter'] = iteration
        
        all_data = pd.concat([all_data,df],ignore_index=True)
    
all_data.to_csv('/home/zach/single_units6.csv')