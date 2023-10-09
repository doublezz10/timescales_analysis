library(R.matlab)
library(tidyverse)
library(minpack.lm)
library(lme4)

set.seed(11061996)

# add big outer loop to loop through datasets

outputs = data.frame(colnames(c('name','area','unit','species','tau','tau_se',
                                'lat','r2','nbSpk')))

data = readMat('/Users/zachz/Library/Mobile Documents/com~apple~CloudDocs/Timescales Data/fontanier_mcc.mat')

spiketimes = data$spikes
cell_info = data$cell.info

for (unit in 1:length(spiketimes)) {
  
  # list of spiketimes in seconds
  spiketimeseries = unlist(lapply(spiketimes[unit], unlist, use.names=FALSE))
  this_cell = unlist(lapply(cell_info[unit], unlist, use.names=FALSE))
  
  species = 'monkey'
  dataset = 'fontanier'
  area = 'mcc'
  
  ## get autocorrelation
  
  curr_ISI = data.frame()
  
  for (lg in 1:100){
    
    # get differences between each spike and the 100 following
    
    ISIs = diff(spiketimeseries,lag=lg)
    ISIs = ISIs[ISIs>0]
    ISIs = ISIs[ISIs<1]
    ISIs = ISIs*1000 # convert to ms
    
    # bin at 33.3ms
    
    bin = hist(ISIs,breaks=seq(from=0,to=1000,length.out=300),plot=FALSE)
    out = data.frame('mids'=bin$mids,'counts'=bin$counts,'lag'=lg)
    
    curr_ISI = rbind(out,curr_ISI)
  }
  
  # sum the ISIs to get the autocorrelation distribution
  
  auto = curr_ISI %>% group_by(mids) %>% 
    summarise(counts=sum(counts)) %>% ungroup()
  
  auto_width = mean(diff(auto$mids))
  
  # remove first 10ms
  
  auto = auto %>% filter(mids>=10)
  
  # convert to probability density fxn
  
  auto_total = sum(auto$counts)
  
  auto = auto %>% mutate(countsn=counts/!!auto_total * 1/!!auto_width)
  
  # smooth it
  
  auto$smooth_countsn = fitted(loess(countsn~mids,data=auto,span=0.1))
  
  # keep only lags up to 5, pdf one more time
  
  curr_ISI = curr_ISI %>% filter(lag<=5)
  
  auto_total = sum(auto$counts)
  
  auto = auto %>% mutate(countsn=counts/!!auto_total * 1/!!auto_width)
  
  # get firing rate
  
  binwidth=0.2
  
  fr = hist(spiketimeseries,
            breaks=seq(from=min(spiketimeseries,na.rm = T),
                       to=max(spiketimeseries,na.rm = T)+binwidth,by=binwidth), 
            plot=FALSE,include.lowest = T)
  
  fr = data.frame("mids"=fr$mids, "counts"=fr$counts)
  
  fr = fr %>% filter(mids != max(mids)) # remove last bin for ceiling effects
  fr <- fr %>% summarise(sd=sd(counts)/binwidth,fr=mean(counts)/
                           binwidth,binwidth=binwidth);
  
  ## now check for multiple peaks
  
  peak.first = which(auto$counts==max(auto$counts))
  if (peak.first==1){
    peak.first = 2
    
    while(!(auto$counts[peak.first-1] < auto$counts[peak.first] & 
            auto$counts[peak.first+1] < auto$counts[peak.first])) {
      peak.first = peak.first+1
    }
  }
  
  peak.first = min(peak.first)
  
  peak.first.lat = min(auto$mids[peak.first])
  peak.first = auto$counts[peak.first]
  
  # check for dip after first peak
  
  post_peak = auto %>% filter(mids >= peak.first.lat)
  
  post_min = min(post_peak$counts) #global min
  post_max = max(post_peak$counts) #global max
  
  # is minimum in first 100ms?
  
  beginning = post_peak %>% filter(between(
    mids,peak.first.lat,peak.first.lat+100))
  
  local_min = min(beginning$counts)
  local_min_lat = min(beginning$mids[beginning$counts==local_min])
  # 'min' will keep only first minimum if there are multiple
  
  # this should be a local minimum, so let's check
  
  local_min_idx = which(post_peak$mids==local_min_lat)
  
  if((post_peak$counts[local_min_idx-1] > post_peak$counts[local_min_idx] & 
      post_peak$counts[local_min_idx+1] > post_peak$counts[local_min_idx]) == T){
    
    # Is the local min less than 75% of the global range?
    
    is_dip <- post_max-local_min >=  3/4 * (post_max - post_min);
    
  } else {
    is_dip <- F;
  }
  
  if(is_dip==TRUE){
    dip.lat <- min(beginning$mids[beginning$counts==local_min]);
    dip <- local_min;
    
    dip_end <- auto %>% filter(mids>dip.lat+12);
    
    peak.second.lat <- min(dip_end$mids[dip_end$counts==max(dip_end$counts)]);
    peak.second  <- max(dip_end$counts);
    
  } else {
    dip.lat <- NA;
    dip <- NA;
    peak.second.lat <- NA;
    peak.second <- NA;
  }
  
  ## Now time for fitting
  
  # here's the fitting fxn expfit
  
  expfit <- function(x,a,b,TAU) {
    a*(exp(-x/TAU))+b;
  }
  
  exp_fit_fun <- function(autocsm,st){
    
    #Fit exponential and extract Tau
    tryCatch({
      
      
      mf <- nlsLM(formula= 'counts ~ A * exp(-mids/TAU) +B ', data=autocsm, start=st);
      
      
      A <- summary(mf)$coefficients[2,1];
      B <- summary(mf)$coefficients[3,1];
      TAU <- summary(mf)$coefficients[1,1];
      TAU.se <- summary(mf)$coefficients[1,2];
      TAU.p <- summary(mf)$coefficients[1,4];
      
      
      autocsm <- autocsm %>% dplyr::mutate(fitval=expfit(mids,A,B,TAU));
      
      fit.err <- sqrt(sum((autocsm$fitval-autocsm$counts)^2));
      
      
      # Return the fit 
      
      df.fit <- data.frame("TAU" = TAU,"TAU.se"=TAU.se,"TAU.p"=TAU.p,"A" = A,"B"=B,"fit.err"=fit.err);
      
    }, error=function(e){
      
      df.fit <- data.frame("TAU" = NA,"TAU.se"=NA,"TAU.p"=NA,"A" = NA,"B"=NA,"fit.err"=NA);
    })
    
    if(!exists("df.fit")){
      df.fit <- data.frame("TAU" = NA,"TAU.se"=NA,"TAU.p"=NA,"A" = NA,"B"=NA,"fit.err"=NA);
    }
    
    
    return(df.fit)
  }
  
  # Now specify starting values
  
  start <- data.frame(TAU=runif(50, min = 0, max = 1000),
                      A=runif(50, min = 0, max = 2*diff(range(auto$counts))),
                      B=runif(50, min = 0, max = 2*min(auto$counts)))
  
  # Fit the first peak
  
  mono = auto %>% filter(mids>=peak.first.lat)
  
  mono_fits = data.frame()
  
  for (fit_iter in 1:nrow(start)){
    
    this_start = start[fit_iter,]
    
    mono_fits = rbind(mono_fits,exp_fit_fun(mono,this_start))
    
  }
  
  mono_fits$dof = nrow(mono)
  mono_fits$fit = 'mono'
  
  mono_fits = mono_fits %>% mutate(rmse = fit.err/sqrt(dof))
  
  all_fits = mono_fits
  
  # fit again, but only from first peak to dip
  
  if(!is.na(dip)){
    
    peak_to_dip = auto %>% filter(between(mids,peak.first.lat,dip.lat))
    
    # do 50 more fits
    
    start <- data.frame(TAU=runif(50, min = 0, max = 1000),
                        A=runif(50, min = 0, 
                                max = 2*diff(range(peak_to_dip$counts))),
                        B=runif(50, min = 0, 
                                max = 2*min(peak_to_dip$counts)))
    
    first_peak_to_dip_fits = data.frame()
    
    for (fit_iter in 1:nrow(start)) {
      
      this_start = start[fit_iter,]
      first_peak_to_dip_fits = rbind(first_peak_to_dip_fits,
                                     exp_fit_fun(peak_to_dip,this_start))
    }
    first_peak_to_dip_fits$fit = 'dip1'
    first_peak_to_dip_fits$dof = nrow(peak_to_dip)
  } else {
    first_peak_to_dip_fits = data.frame("TAU" = NA,"TAU.se"=NA,"TAU.p"=NA,
                                        "A" = NA,"B"=NA,"fit.err"=NA,
                                        fit="dip1",dof=NA);
  }
  
  
  
  first_peak_to_dip_fits = first_peak_to_dip_fits %>% mutate(rmse = fit.err/sqrt(dof))
  
  all_fits = rbind(all_fits,first_peak_to_dip_fits)
  
  # repeat for second peak
  
  if(!is.na(peak.second)){
    
    second_peak_to_end = auto %>% filter(mids>=peak.second.lat)
    
    # do 50 more fits
    
    start <- data.frame(TAU=runif(50, min = 0, max = 1000),
                        A=runif(50, min = 0, 
                                max = 2*diff(range(second_peak_to_end$counts))),
                        B=runif(50, min = 0, 
                                max = 2*min(second_peak_to_end$counts)))
    
    second_peak_to_end_fits = data.frame()
    
    for (fit_iter in 1:nrow(start)){
      
      this_start = start[fit_iter,]
      second_peak_to_end_fits = rbind(second_peak_to_end_fits,
                                      exp_fit_fun(second_peak_to_end,this_start))
    }
    second_peak_to_end_fits$fit = 'dip2'
    second_peak_to_end_fits$dof = nrow(second_peak_to_end)
    
    
    second_peak_to_end_fits = second_peak_to_end_fits %>% mutate(rmse = fit.err/sqrt(dof))
    
    all_fits = rbind(all_fits,second_peak_to_end_fits)
    
  }
  
  all_fits$peak.lat = peak.first.lat
  all_fits$peak = peak.first
  all_fits$dip.lat = dip.lat
  all_fits$dip = dip
  all_fits$peak.second.lat = peak.second.lat
  all_fits$peak.second = peak.second
  
  ## filter for valid fits only
  
  good_fits = all_fits %>% filter(TAU<1000 & TAU > 0 & A>0 & B>0)
  
  if (nrow(good_fits) == 0){
    
    result = data.frame('name'=dataset,'area'=area,'unit'=unit,'species'=species,
                        'tau'=NA,'tau.se'=NA,
                        'lat'=NA,'r2'=NA,
                        'nbSpk'=length(spiketimeseries))
    
    outputs = rbind(outputs,result)
    
  } else{
    
    # now keep best one
    
    best_fit = good_fits %>% slice_min(rmse)
    
    best_fit = best_fit %>% mutate(r2 = 1-(rmse^2/fit.err^2))
    
    best_fit = best_fit[1,]
    
    result = data.frame('name'=dataset,'area'=area,'unit'=unit,'species'=species,
                        'tau'=best_fit$TAU,'tau.se'=best_fit$TAU.se,
                        'lat'=best_fit$peak.lat,'r2'=best_fit$r2,
                        'nbSpk'=length(spiketimeseries),'fr'=fr$fr)
    
    
    outputs = rbind(outputs,result)
    
  }
 
  
  
  
}
