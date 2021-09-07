#%%
 
import numpy as np
import pandas as pd

individual_data = pd.read_csv('/Users/zachz/Documents/timescales_analysis/1000iter results/by_individual/by_individual.csv')
listofspecies = ['mouse','rat','monkey','human']

individual_data['species'] = pd.Categorical(individual_data['species'], categories = listofspecies , ordered = True)

# %%

# filter for r2 and tau criteria

# get mean across iterations

# then plot separated by individual (color?)
# or each graph is a dataset, col is brain_area, and x = individual