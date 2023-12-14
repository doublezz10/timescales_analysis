#%%

# imports

import numpy as np
import scipy.io as spio
import pandas as pd

from scipy.optimize import curve_fit

from scipy.signal import savgol_filter

import matplotlib.pyplot as plt

import os

import mat73

pd.options.mode.chained_assignment = None

ground_truth = pd.read_csv('/Users/zachz/Documents/timescales_analysis/1000iter results/isi figs/filtered_isi_data.csv')

#%%

directory = '/Users/zachz/Documents/timescales_analysis/2023/python_diagnosis'

all_data = pd.DataFrame()
all_mids = []
all_fitvals = []
all_auto = []

for filename in sorted(os.listdir(directory)):
        
    if filename.endswith('.mat'):
        
        this_dataset = pd.DataFrame()

        f = os.path.join(directory, filename)
        
        items = filename.split('_')
        
        dataset = items[0]
        area = items[1].replace('.mat','')

        try:
            data_dict = mat73.loadmat(f)
        except:
            data_dict = spio.loadmat(f,squeeze_me=True,simplify_cells=True)

        spiketimes = data_dict['spikes']
        
        print(dataset,area)

        for unit in range(10):
            
            print('now fitting unit %i out of %i' %(unit,len(spiketimes)))
            
            spiketimeseries = spiketimes[unit]

            # do autocorrelation

            curr_isi = pd.DataFrame()

            for lag in range(101):
                
                # get differences bw each spike and the 100 following
                
                isi = spiketimeseries[lag+1:] - spiketimeseries[1:(len(spiketimeseries)-lag)]
                isi = isi[isi>0]
                isi = isi[isi<1]
                isi = isi*1000 # convert to ms
                
                # bin at 3.33ms
                
                binned = np.histogram(isi,np.linspace(0,1000,300))
                mids = binned[1][:-1] + np.diff(binned[1])/2
                
                out = pd.DataFrame({'mids':mids,'counts':binned[0],'lag':np.ones_like(mids)*lag+1})
                
                curr_isi = pd.concat((curr_isi,out))

            # sum the ISIs to get the autocorrelation

            auto = curr_isi.groupby(by='mids',as_index=False).sum()

            
            auto = auto.drop(columns=['lag'])

            auto_width = np.mean(np.diff(auto['mids']))

            # remove first 10ms

            auto = auto[auto.mids>=10].reset_index()

            # make probability density fxn

            auto_total = np.sum(auto['counts'])

            auto['countsn'] = (auto['counts']/auto_total) * (1/(1/300))

            # smooth

            auto['smooth_countsn'] = savgol_filter(auto['countsn'],window_length=99,polyorder=2)

            # keep only lags up to 5, fit pdf one more time

            ISIs = curr_isi[curr_isi['lag']<=5]
            ISIs_tot = np.sum(ISIs['counts'])

            ISIs['countsn'] = ISIs['counts']/ISIs_tot * (1/(1/300))

            # get fr

            binwidth = 0.2

            fr = np.histogram(spiketimeseries,bins=np.arange(np.nanmin(spiketimeseries),np.nanmax(spiketimeseries),step=binwidth))

            fr = pd.DataFrame({'mids': fr[1][:-1] + np.diff(fr[1])/2,'counts':fr[0]})

            fr = fr[fr['mids'] != np.nanmax(fr['mids'])] # remove last bin for ceiling effects

            fr['sd'] = np.std(fr['counts'])/binwidth
            fr['mean'] = np.mean(fr['counts']/binwidth)
            fr['binwidth'] = binwidth

            # now time for careful peak detection

            peak_first = np.argmax(auto['counts'])

            if peak_first == 0:
                peak_first = 1
                
                while auto.counts[peak_first-1] < auto.counts[peak_first] and auto.counts[peak_first + 1] < auto.counts[peak_first]:
                    peak_first = peak_first+1
                    
            peak_first = np.min(peak_first) # in case multiple

            peak_first_lat = np.min(auto.mids.iloc[peak_first])
            peak_first = auto.counts.iloc[peak_first]

            # now check for dip after this peak

            post_peak = auto[auto.mids >= peak_first_lat]

            post_min = np.nanmin(post_peak.counts)
            post_max = np.nanmax(post_peak.counts)

            # is min in first 100ms

            beginning = post_peak[np.logical_and(post_peak.mids > peak_first_lat, post_peak.mids < peak_first_lat + 100)]

            local_min = np.nanmin(beginning.counts)
            local_min_lat = np.nanmin(beginning.mids[beginning.counts==local_min])

            # check that this is actually a local min

            local_min_idx = np.where(post_peak.mids==local_min_lat)[0][0]

            try:
                if np.logical_and(post_peak.counts.iloc[local_min_idx-1] > post_peak.counts.iloc[local_min_idx], post_peak.counts.iloc[local_min_idx+1] > post_peak.counts.iloc[local_min_idx]):
                
                    is_dip = post_max - local_min >= .75 * (post_max-post_min)
                
                else:
                    is_dip = False
            
            except IndexError:
                is_dip = False

            if is_dip == True:
                
                dip_lat = np.nanmin(beginning.mids.iloc[np.where(beginning.counts==local_min)])
                dip = local_min
                
                dip_end = auto[auto.mids>dip_lat+12]
                
                peak_second_lat = np.nanmin(dip_end.mids.iloc[np.where(dip_end.counts==np.nanmax(dip_end.counts))])
                peak_second = np.nanmax(dip_end.counts)
                
            else:
                dip_lat = np.nan
                dip = np.nan
                peak_second_lat = np.nan
                peak_second = np.nan
                
            def func(x,a,tau,b):
                return a*((np.exp(-x/tau))+b)

            def curve_fit_func(auto,start):
                
                counts = auto['smooth_countsn']
                mids = auto['mids']
                
                try:
                    pars,cov = curve_fit(func,mids,counts,p0=[1,100,1],bounds=((0,np.inf)),maxfev=5000)

                    a = pars[0]
                    tau = pars[1]
                    b = pars[2]
                    
                    perr = np.sqrt(np.diag(cov))
                    
                    tau_err = perr[1]
                    
                    auto['fitvals'] = func(auto['counts'],*pars)
                    
                    fit_err = np.sqrt(np.sum(auto['fitvals']-auto['counts'])**2)
                    r2 = np.corrcoef(auto['fitvals'],auto['counts'])[1,0] * -1
                    
                    # store the fit
                    
                    df_fit = pd.DataFrame({'tau':[tau],'tau_error':[tau_err],'A':[a],'B':[b],'fit_err':[fit_err],'r2':[r2]})           
                except:
                    
                    df_fit = pd.DataFrame({'tau':[np.nan],'tau_error':[np.nan],'A':[np.nan],'B':[np.nan],'fit_err':[np.nan],'r2':[np.nan]})
                    
                return(df_fit)

            # generate some starting values

            starts = pd.DataFrame({'taus':np.random.choice(np.arange(0,1000,0.1),50),'A':np.random.choice(np.arange(0,20,0.1),50),'B':np.random.choice(np.arange(0,20,0.1),50)})

            # first peak fitting :)

            mono = auto[auto.mids >= peak_first_lat]

            mono_fits = pd.DataFrame()

            for iter in range(len(starts)):
                
                this_start = starts.iloc[iter]
                
                mono_fits = pd.concat((mono_fits,curve_fit_func(mono,this_start)))
                
            mono_fits['dof'] = len(mono)
            mono_fits['fit'] = 'mono'

            mono_fits['rmse'] = mono_fits['fit_err']/np.sqrt(mono_fits['dof'])

            all_fits = mono_fits

            # now fit only to first dip

            if is_dip == True:
                
                peak_to_dip = auto[np.logical_and(auto.mids>peak_first_lat,auto.mids<dip_lat)]
                                
                starts = pd.DataFrame({'taus':np.random.choice(np.arange(0,1000,0.1),50),'A':np.random.choice(np.arange(0,100,0.1),50),'B':np.random.choice(np.arange(0,100,0.1),50)})

                first_peak_to_dip_fits = pd.DataFrame()
                
                for iter in range(len(starts)):
                    
                    this_start = starts.iloc[iter]
                    
                    first_peak_to_dip_fits = pd.concat((first_peak_to_dip_fits,curve_fit_func(peak_to_dip,this_start)))
                    
                first_peak_to_dip_fits['fit'] = 'dip1'
                first_peak_to_dip_fits['dof'] = len(peak_to_dip)
                
            else:
                first_peak_to_dip_fits = pd.DataFrame({'tau':[np.nan],'tau_error':[np.nan],'A':[np.nan],'B':[np.nan],'fit_err':[np.nan],'fit':['dip1'],'dof':[np.nan]})

            all_fits = pd.concat((all_fits,first_peak_to_dip_fits))

            # second peak now

            if is_dip == True:
                
                second_peak_to_end = auto[auto.mids>=peak_second_lat]
                
                # 50 more :)
                
                starts = pd.DataFrame({'taus':np.random.choice(np.arange(0,1000,0.1),50),'A':np.random.choice(np.arange(0,100,0.1),50),'B':np.random.choice(np.arange(0,100,0.1),50)})
                
                second_peak_to_end_fits = pd.DataFrame()
                
                for iter in range(len(starts)):
                    
                    this_start = starts.iloc[iter]
                    
                    second_peak_to_end_fits = pd.concat((second_peak_to_end_fits,curve_fit_func(second_peak_to_end,this_start)))
                    
                second_peak_to_end_fits['fit'] = 'dip2'
                second_peak_to_end_fits['dof'] = len(second_peak_to_end)
                
                second_peak_to_end_fits['rmse'] = second_peak_to_end_fits['fit_err'] / np.sqrt(second_peak_to_end_fits['dof'])
                
                all_fits = pd.concat((all_fits,second_peak_to_end_fits))
                
            all_fits['peak_lat'] = peak_first_lat
            all_fits['peak'] = peak_first
            all_fits['dip_lat'] = dip_lat
            all_fits['dip'] = dip
            all_fits['peak_second_lat'] = peak_second_lat
            all_fits['peak_second'] = peak_second
            all_fits['nbSpk'] = len(spiketimeseries)
            
            all_fits['unit'] = unit
            
            all_fits['dataset'] = dataset
            all_fits['area'] = area
            
            all_fits['fr'] = fr['mean']
            
            best_fit = pd.DataFrame(all_fits[all_fits.rmse == np.nanmin(all_fits.rmse)])
            
            if len(best_fit) > 1:
                
                best_fit = pd.DataFrame(best_fit.iloc[0]).T
                
            mids = auto['mids'].to_numpy()
            
            this_font_unit = ground_truth.iloc[unit]
            
            zach_fit = func(mids,float(best_fit.A),float(best_fit.tau),float(best_fit.B))            
            real_auto = auto['countsn']
                    
            fred_fit = func(mids,float(this_font_unit.A),float(this_font_unit.tau),float(this_font_unit.B))
                    
            plt.plot(mids,zach_fit,label='zach norm')
            plt.plot(mids,real_auto,label='real dist')            
            plt.title('tau = %.2f, lat = %.2f' %(best_fit.tau,best_fit.peak_lat))
            
            plt.xlabel('time (ms)')
            plt.ylabel('autocorrelation')
            
            plt.legend()
            plt.show()
            
            all_mids.append(mids)
            all_fitvals.append(func(mids,float(best_fit.A),float(best_fit.tau),float(best_fit.B)))
            all_auto.append(auto['smooth_countsn'])
            
            this_dataset = pd.concat((this_dataset,best_fit))
            
            
        all_data = pd.concat((all_data,this_dataset))
# %%

