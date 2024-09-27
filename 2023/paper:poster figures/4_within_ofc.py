#%%

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

import statsmodels.api as sm


import statsmodels.formula.api as smf


from scipy import interpolate

plt.style.use('seaborn')

data = pd.read_csv('final_data.csv')

data['species'] = pd.Categorical(data['species'], categories = ['mouse','monkey','human'] , ordered = True)

filt_data = data[np.logical_and(data.tau < 1000, data.r2 > 0.5)]
filt_data = filt_data[filt_data.tau > 10]

fred_data = filt_data[filt_data.name=='stoll']

ofc_lai_vl = pd.DataFrame()

for area_ in ['LAI','a11l','a11m','a12l','a12m','a12o','a12r','a13l','a13m']:
    
    ofc_lai_vl = pd.concat((ofc_lai_vl,fred_data[fred_data.area==area_]))

ofc_lai_vl['area'] = ofc_lai_vl['area'].str.replace('LAI','AI')

ofc_lai_vl['area'] = ofc_lai_vl['area'].str.replace('11m','11m/l')
ofc_lai_vl['area'] = ofc_lai_vl['area'].str.replace('11l','11m/l')

ofc_lai_vl.loc[ofc_lai_vl['area'] == 'a11m/l', 'granularity'] = 'granular'
ofc_lai_vl.loc[ofc_lai_vl['area']== 'a12m', 'granularity'] = 'granular'
ofc_lai_vl.loc[ofc_lai_vl['area']== 'a12l', 'granularity'] = 'granular'
ofc_lai_vl.loc[ofc_lai_vl['area']== 'a12o', 'granularity'] = 'granular'
ofc_lai_vl.loc[ofc_lai_vl['area']== 'a12r', 'granularity'] = 'dysgranular'
ofc_lai_vl.loc[ofc_lai_vl['area'] =='a13m', 'granularity'] = 'dysgranular'
ofc_lai_vl.loc[ofc_lai_vl['area'] == 'a13l', 'granularity'] = 'dysgranular'
ofc_lai_vl.loc[ofc_lai_vl['area'] =='AI', 'granularity'] = 'agranular'

ofc_lai_vl['granularity'] = pd.Categorical(ofc_lai_vl['granularity'], categories=['granular','dysgranular','agranular'], ordered=True)

counts = ofc_lai_vl.groupby('area',as_index=False).size()

counts.loc[counts['area'] == 'a11m/l', 'granularity'] = 'granular'
counts.loc[counts['area']== 'a12m', 'granularity'] = 'granular'
counts.loc[counts['area']== 'a12l', 'granularity'] = 'granular'
counts.loc[counts['area']== 'a12o', 'granularity'] = 'granular'
counts.loc[counts['area']== 'a12r', 'granularity'] = 'dysgranular'
counts.loc[counts['area'] =='a13m', 'granularity'] = 'dysgranular'
counts.loc[counts['area'] == 'a13l', 'granularity'] = 'dysgranular'
counts.loc[counts['area'] =='AI', 'granularity'] = 'agranular'

counts['granularity'] = pd.Categorical(counts['granularity'], categories=['granular','dysgranular','agranular'], ordered=True)

# repeat on unfiltered data

ofc_lai_vl2 = pd.DataFrame()

for area_ in ['LAI','a11l','a11m','a12l','a12m','a12o','a12r','a13l','a13m']:
    
    ofc_lai_vl2 = pd.concat((ofc_lai_vl2,data[data.area==area_]))

ofc_lai_vl2['area'] = ofc_lai_vl2['area'].str.replace('LAI','AI')

ofc_lai_vl2['area'] = ofc_lai_vl2['area'].str.replace('11m','11m/l')
ofc_lai_vl2['area'] = ofc_lai_vl2['area'].str.replace('11l','11m/l')

ofc_lai_vl2.loc[ofc_lai_vl2['area'] == 'a11m/l', 'granularity'] = 'granular'
ofc_lai_vl2.loc[ofc_lai_vl2['area']== 'a12m', 'granularity'] = 'granular'
ofc_lai_vl2.loc[ofc_lai_vl2['area']== 'a12l', 'granularity'] = 'granular'
ofc_lai_vl2.loc[ofc_lai_vl2['area']== 'a12o', 'granularity'] = 'granular'
ofc_lai_vl2.loc[ofc_lai_vl2['area']== 'a12r', 'granularity'] = 'dysgranular'
ofc_lai_vl2.loc[ofc_lai_vl2['area'] =='a13m', 'granularity'] = 'dysgranular'
ofc_lai_vl2.loc[ofc_lai_vl2['area'] == 'a13l', 'granularity'] = 'dysgranular'
ofc_lai_vl2.loc[ofc_lai_vl2['area'] =='AI', 'granularity'] = 'agranular'

ofc_lai_vl2['granularity'] = pd.Categorical(ofc_lai_vl2['granularity'], categories=['granular','dysgranular','agranular'], ordered=True)

counts2 = ofc_lai_vl2.groupby('area',as_index=False).size()

counts2.loc[counts2['area'] == 'a11m/l', 'granularity'] = 'granular'
counts2.loc[counts2['area']== 'a12m', 'granularity'] = 'granular'
counts2.loc[counts2['area']== 'a12l', 'granularity'] = 'granular'
counts2.loc[counts2['area']== 'a12o', 'granularity'] = 'granular'
counts2.loc[counts2['area']== 'a12r', 'granularity'] = 'dysgranular'
counts2.loc[counts2['area'] =='a13m', 'granularity'] = 'dysgranular'
counts2.loc[counts2['area'] == 'a13l', 'granularity'] = 'dysgranular'
counts2.loc[counts2['area'] =='AI', 'granularity'] = 'agranular'

counts2['granularity'] = pd.Categorical(counts2['granularity'], categories=['granular','dysgranular','agranular'], ordered=True)


#%%

fig,axs = plt.subplots(1,1,figsize=(3.4,2))

sns.barplot(ax=axs,data=counts,x='area',y='size',hue='granularity',palette='Set2',dodge=False)
sns.barplot(ax=axs,data=counts2,x='area',y='size',hue='granularity',palette='Set2',alpha=0.5,dodge=False)

axs.set_xlabel(None)
axs.set_xticks(range(8),['AI','11m/l','12l','12m','12o','12r','13l','13m'])
axs.tick_params(axis='x',rotation=0, labelsize=7)
axs.tick_params(axis='y',labelsize=7)
axs.set_ylabel('number of neurons',fontsize=7)
axs.legend(title='',prop={'size':7})

plt.show()

#%%

fig, axs = plt.subplots(1,2,figsize=(7,3))

sns.pointplot(ax=axs[0],data=ofc_lai_vl[ofc_lai_vl.area != '13b'],x='area',y='tau',hue='granularity',ci=95,estimator=np.mean,palette="Set2",linestyle='',join=False,errwidth=1.5)

axs[0].set_xlabel(None)
axs[0].set_xticks(range(8),['AI','11m/l','12l','12m','12o','12r','13l','13m'])
axs[0].tick_params(axis='x', labelsize=7)
axs[0].tick_params(axis='y',labelsize=7)
axs[0].set_ylabel('timescale (ms)',fontsize=7)


sns.pointplot(ax=axs[1],data=ofc_lai_vl[ofc_lai_vl.area != '13b'],x='area',y='lat',hue='granularity',ci=95,estimator=np.mean,palette="Set2",linestyle='',join=False,errwidth=1.5)

axs[1].set_xlabel(None)
axs[1].set_xticks(range(8),['AI','11m/l','12l','12m','12o','12r','13l','13m'])
axs[1].tick_params(axis='x',labelsize=7)
axs[1].tick_params(axis='y',labelsize=7)
axs[1].set_ylabel('latency (ms)',fontsize=7)

plt.tight_layout()

plt.show()
#%%

for animal in ofc_lai_vl.animID.unique():
    
    this_animal = ofc_lai_vl[ofc_lai_vl.animID==animal]

    ofc_lai_vl_ = np.around(this_animal,0)

    ys = ofc_lai_vl_.AP.to_numpy()
    xs = ofc_lai_vl_.ML.to_numpy()
    zs = ofc_lai_vl_.tau.to_numpy()

    grouped_df = ofc_lai_vl_.groupby(['AP', 'ML'],as_index=False
                            ).mean()

    grouped_df['count'] = ofc_lai_vl_.groupby(['AP','ML']).count().unitID.values

    avg = ofc_lai_vl_

    points = np.column_stack((xs,ys))

    grid_xn, grid_yn = np.mgrid[np.min(xs):np.max(xs):32j, np.min(ys):np.max(ys):26j]

    f = interpolate.LinearNDInterpolator(points, zs)
    zn = f(grid_xn, grid_yn)
    
    plt.style.use('seaborn')

    plt.figure(figsize=(5,4))

    plt.pcolor(grid_xn,grid_yn,zn,shading='auto',cmap='Greys',vmin=0,vmax=1000)
    plt.xlabel('ML (mm)',fontsize=7)
    plt.ylabel('AP (mm)',fontsize=7)
    plt.xticks(fontsize=7)
    plt.yticks(fontsize=7)

    cbar = plt.colorbar()
    cbar.set_label(label='timescale (ms)',size=7)
    cbar.ax.set_yticklabels([0,200,400,600,800,1000],fontsize=7)

    #sns.scatterplot(data=grouped_df,x='ML',y='AP',hue='tau',palette='viridis',hue_norm=(0,1000),legend=True)

    #plt.legend(bbox_to_anchor=(-0.2, 1.0),loc='upper right',prop={'size':7})

    #sns.scatterplot(x='ML',y='AP',hue='area',data=this_animal,alpha=0.2)
    plt.legend(bbox_to_anchor=(-0.5, 1.0), loc='upper left',prop={'size':7})
    if animal == 1:
        plt.gca().invert_xaxis()
        
        plt.savefig('mk2ofctau.svg',dpi=450)
    else: 
        plt.savefig('mk1ofctau.svg',dpi=450)

    plt.show()
    
    plt.style.use('default')
    
    plt.figure(figsize=(4.3,2))
    sns.lineplot(data=ofc_lai_vl_,x='ML',y='tau')
    if animal == 1:
        plt.savefig('mk2_ml.svg',dpi=450)
        plt.gca().invert_xaxis()
    else:
        plt.savefig('mk1_ml.svg',dpi=450)
    plt.show()
    
    plt.figure(figsize=(4.3,2))
    sns.lineplot(data=ofc_lai_vl_,x='AP',y='tau')
    if animal == 1:
        plt.savefig('mk2_ap.svg',dpi=450)
    else:
        plt.savefig('mk1_ap.svg',dpi=450)
    plt.show()

#%%

model = smf.ols('tau ~ AP * ML',data=avg)

res = model.fit()

anova = sm.stats.anova_lm(res, typ=1)

anova

# %%

ofc_lai_vl = np.around(ofc_lai_vl,0)

ys = ofc_lai_vl.AP.to_numpy()
xs = ofc_lai_vl.ML.to_numpy()
zs = ofc_lai_vl.lat.to_numpy()

grouped_df = ofc_lai_vl.groupby(['AP', 'ML'],as_index=False
                          ).mean()

grouped_df['count'] = ofc_lai_vl.groupby(['AP','ML']).count().unitID.values

avg = ofc_lai_vl

points = np.column_stack((xs,ys))

grid_xn, grid_yn = np.mgrid[np.min(xs):np.max(xs):20j, np.min(ys):np.max(ys):20j]

f = interpolate.LinearNDInterpolator(points, zs)
zn = f(grid_xn, grid_yn)

plt.figure(figsize=(5,4))

plt.pcolor(grid_xn,grid_yn,zn,shading='auto',cmap='Greys',vmin=0,vmax=1000)
plt.xlabel('ML (mm)',fontsize=7)
plt.ylabel('AP (mm)',fontsize=7)
plt.xticks(fontsize=7)
plt.yticks(fontsize=7)

cbar = plt.colorbar()
cbar.set_label(label='latency (ms)',size=7)
cbar.ax.set_yticklabels([0,200,400,600,800,1000],fontsize=7)

sns.scatterplot(data=grouped_df,x='ML',y='AP',hue='lat',palette='inferno',hue_norm=(0,200),legend=True)

plt.legend(bbox_to_anchor=(-0.2, 1.0),loc='upper right',prop={'size':7})

# sns.scatterplot(x='ML',y='AP',hue='specific_area',data=ofc_lai_vl,alpha=0.2)
# plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')

plt.gca().invert_xaxis()

plt.show()


#%%

animal = 1
    
this_animal = ofc_lai_vl[ofc_lai_vl.animID==animal]

ofc_lai_vl_ = np.around(this_animal,0)

model = smf.ols('lat ~ AP * ML',data=ofc_lai_vl_)

res = model.fit()
print(animal)
print(res.summary())

ys = ofc_lai_vl_.AP.to_numpy()
xs = ofc_lai_vl_.ML.to_numpy()

grid_xn, grid_yn = np.mgrid[np.min(xs):np.max(xs):32j, np.min(ys):np.max(ys):26j]

fitvals = np.reshape(res.predict(exog=pd.DataFrame({'AP':np.hstack(grid_yn),'ML':np.hstack(grid_xn)})).values,(32,26))


plt.figure(figsize=(4,3.2))
plt.pcolor(grid_xn,grid_yn,fitvals,shading='auto',cmap='Greys',vmin=25,vmax=105)
sns.scatterplot(x='ML',y='AP',hue='area',data=this_animal,alpha=0.2)
plt.gca().invert_xaxis()
cbar = plt.colorbar()
cbar.set_label(label='latency (ms)',size=7)
cbar.ax.set_yticks([25,45,65,85,105])
cbar.ax.set_yticklabels([25,45,65,85,105],fontsize=7)
plt.xlabel('ML (mm)',fontsize=7)
plt.ylabel('AP (mm)',fontsize=7)
plt.xticks(fontsize=7)
plt.yticks(fontsize=7)

plt.show()

#%%

animal = 2
    
this_animal = ofc_lai_vl[ofc_lai_vl.animID==animal]

ofc_lai_vl_ = np.around(this_animal,0)

model = smf.ols('lat ~ AP * ML',data=ofc_lai_vl_)

res = model.fit()
print(animal)
print(res.summary())

ys = ofc_lai_vl_.AP.to_numpy()
xs = ofc_lai_vl_.ML.to_numpy()

grid_xn, grid_yn = np.mgrid[np.min(xs):np.max(xs):32j, np.min(ys):np.max(ys):26j]

fitvals = np.reshape(res.predict(exog=pd.DataFrame({'AP':np.hstack(grid_yn),'ML':np.hstack(grid_xn)})).values,(32,26))


plt.figure(figsize=(4,3.2))
plt.pcolor(grid_xn,grid_yn,fitvals,shading='auto',cmap='Greys',vmin=25,vmax=105)
sns.scatterplot(x='ML',y='AP',hue='area',data=this_animal,alpha=0.2)
cbar = plt.colorbar()
cbar.set_label(label='timescale (ms)',size=7)
cbar.ax.set_yticks([25,45,65,85,105])
cbar.ax.set_yticklabels([25,45,65,85,105],fontsize=7)
plt.xlabel('ML (mm)',fontsize=7)
plt.ylabel('AP (mm)',fontsize=7)
plt.xticks(fontsize=7)
plt.yticks(fontsize=7)
plt.show()

#%%



# %%


