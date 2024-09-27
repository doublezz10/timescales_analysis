function [out] = ISI_timescales(timestamps)

%- param for fit
nPerm = 100;
A_test = 0:0.01:20;
B_test = 0:0.01:20;
Tau_test = 0:0.1:1000;
maxFailure = 100;

if ~isempty(timestamps)
    if size(timestamps,2)==1
        timestamps = timestamps';
    end
end

bins_hist = [0:.0033:1]; % bins
FAILED = false;
if ~isempty(timestamps)
    %- Time difference with lags up to 100
    data_all=[];
    for i = 1 : 100
        data_all = [data_all timestamps(i+1:end)-timestamps(1:end-i)]; %- same results than function diff in R
    end
    
    %- remove diff below 0 and above 1 sec
    data_all = data_all(data_all>0 & data_all<1);
    
    %- binning
    [data,bins] = hist(data_all,bins_hist); % 3.33ms bins
    
    %- minmax normalization
    data_norm=(data-min(data))/(max(data)-min(data));
    
    %- loess smooth
    data_norm_smoothed = smooth(bins,data_norm,0.1,'loess'); %- 0.1 = 10% for Vincent
    %data_norm_smoothed = data_norm';
    
    %- find the peak (improve that) and remove points before
    start = find(data_norm_smoothed==max(data_norm_smoothed),1,'first');
    data_short_norm = data_norm_smoothed(start:end);
    bins_short = bins(start:end)*1000; %- convert in ms
    
    %- exponential decay fit function : R(delta_K) = A[exp(-delta_time_lag/time_constant)+B]
    g = fittype('A*(exp(-x/time_constant)+B)');
    
    %- perform the fit 50 times with randomly selected starting points (see params on top of script)
    %- and keep the one with lowest error
    clear model_error models
    h=0; nbFailed=0;
    disp('Fitting in progress')
    while h < nPerm && nbFailed<maxFailure %- skip if fits failed 500 times
        try
            [f0,gof] = fit(bins_short',data_short_norm,g,...
                'StartPoint',[A_test(randperm(length(A_test),1))...
                Tau_test(randperm(length(Tau_test),1))...
                B_test(randperm(length(B_test),1))]); % could use 'Lower' and 'Upper' to constrained a bit the model...
            h = h + 1; % disp(h)
            model_error(h) = gof.sse; %- extract fit error
            models(h).f0 = f0; %- extract fitting params
            model_r(h) = gof.adjrsquare; %- extract r square
        catch
            nbFailed = nbFailed + 1;
        end
    end
    
    
    if nbFailed~=maxFailure
        disp('Fitting done')
        %- find best model (lowest error) and keep that one
        best_model = find(model_error==min(model_error));
        f0 = models(best_model(1)).f0;
        sse = model_error(best_model(1));
        r2 = model_r(best_model(1));
        
        % plot it
            fig = figure;
            plot(bins*1000,data_norm,'Color',[.8 .8 .8]);hold on
            plot(bins*1000,data_norm_smoothed,'Color','k');hold on
            plot(bins_short(1),data_short_norm(1),'.k','MarkerSize',10)
        
            hold on;
            plot(f0,'predfunc',.95);legend off;
            xlim([0 800]);ylim([0 1])
            xlabel('time (ms)')
            ylabel('autocorrelation (a.u.)')
            text(350,.9,['nSPK=' num2str(length(timestamps))],'FontSize',5)
            text(350,.8,['lat=' num2str(bins_short(1))],'FontSize',5)
            text(350,.7,['Tau=' num2str(round(f0.time_constant))],'FontSize',5)
        
        
            disp(f0)
            disp(nbFailed)
        
            pause
            %close(fig)
        
        out.bins_short = bins_short;
        out.bins = bins;
        out.data_norm = data_norm;
        out.data_norm_smoothed = data_norm_smoothed;
        out.data_short_norm = data_short_norm;
        out.tau = f0.time_constant;
        out.lat = bins_short(1);
        out.f0 = f0;
        out.sse = sse;
        out.r2 = r2;
        
    else
        disp('Fitting failed')
        FAILED = true;
    end
    
else %- if no timestamps
    
    FAILED = true;

end


if FAILED
    out.bins_short = [];
    out.bins = [];
    out.data_norm = [];
    out.data_norm_smoothed = [];
    out.data_short_norm = [];
    out.tau = [];
    out.lat = [];
    out.f0 = [];
    out.sse = [];
    out.r2 = [];
end





