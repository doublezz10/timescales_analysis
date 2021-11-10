# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 15:06:41 2021

@author: zachz
"""

import pandas as pd

from single_neuron_timescales import get_single_unit_timescales

files = [('/home/zach/Meg - monkey/meg_amygdala.mat','meg','monkey','amygdala'),
        ('/home/zach/Meg - monkey/meg_scACC.mat','meg','monkey','scACC'),
        ('/home/zach/Meg - monkey/meg_ventralStriatum.mat','meg','monkey','vStriatum'),
        ('/home/zach/Minxha - Human MFC/minxha_amygdala.mat','minxha','human','amygdala'),
        ('/home/zach/Minxha - Human MFC/minxha_dACC.mat','minxha','human','dACC'),
        ('/home/zach/Minxha - Human MFC/minxha_hippocampus.mat','human','minxha','hc'),
        ('/home/zach/Minxha - Human MFC/minxha_preSMA.mat','minxha','human','preSMA'),
        ('/home/zach/Peyrache - rat mPFC/peyrache_mPFC.mat','peyrache','rat','mpfc'),
        ('/home/zach/Steinmetz - mouse/stein_aca.mat','steinmetz','mouse','aca'),
        ('/home/zach/Steinmetz - mouse/stein_bla.mat','steinmetz','mouse','bla'),
        ('/home/zach/Steinmetz - mouse/stein_ca1.mat','steinmetz','mouse','ca1'),
        ('/home/zach/Steinmetz - mouse/stein_ca2.mat','steinmetz','mouse','ca2'),
        ('/home/zach/Steinmetz - mouse/stein_ca3.mat','steinmetz','mouse','ca3'),
        ('/home/zach/Steinmetz - mouse/stein_dg.mat','steinmetz','mouse','dg'),
        ('/home/zach/Steinmetz - mouse/stein_ila.mat','steinmetz','mouse','ila')
        ]

all_data = pd.DataFrame()

for file in range(len(files)):
    
    for iteration in range(1000):
    
        print('Iteration #', iteration)

        df = get_single_unit_timescales(files[file][0],files[file][1],files[file][2],files[file][3])
        
        df['iter'] = iteration
        
        all_data = pd.concat([all_data,df],ignore_index=True)
    
all_data.to_csv('/home/zach/single_units2.csv')