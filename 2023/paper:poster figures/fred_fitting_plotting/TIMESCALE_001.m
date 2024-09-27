clear

%- param for fit
nTau = 1; %50 %- put 1 if you want to do it on all the timestamps!
nSpk = 1000;

path2go = '/Users/zachz/Library/Mobile Documents/com~apple~CloudDocs/Timescales Raw Data/';

list = dir([path2go '*.mat']);

%%

for d = 1 : length(list)
    if strncmp(list(d).name,'stoll_AMG',9) == 1
        
        %- load dataset file
        load([path2go list(d).name])
        disp(list(d).name)
        
        %- loop on neurons
        clear out
        
        for p = 1 : nTau
            for n = 10 : length(spikes)
                disp(n)
                if nTau>1 %- if you want to do it on subsamples
                    if length(spikes{n}) > nSpk
                        %- take a subset of all spike times
                        rdperm = randperm(length(spikes{n}));
                        [out{n,p}] = ISI_timescales(sort(spikes{n}(rdperm(1:nSpk))));
                    else
                        [out{n,p}] = ISI_timescales([]);
                        
                    end
                else
                    [out{n,p}] = ISI_timescales(spikes{n})
                end
            end
        end
        
        if nTau==1
            out = out';
        end
        %- save output
        save([path2go '/' fol '/ISI_Timescales_' list(d).name(1:end-4) '.mat'],'out','spikes')
        clear out spikes
    end
end


%- that worked 1 time for some reason but now it doesn't like it...?!?!?!
% subsp = true;
%
% for d = 1 : length(list)
%     if exist([path2go '/processed/ISI_Timescales_' list(d).name(1:end-4) '.mat'])==0
%         
%         %- load dataset file
%         load([path2go list(d).name])
%         disp(list(d).name)
%         
%         %- loop on neurons
%         clear out
%         
%         
%         parfor n = 1 : length(spikes)
%             disp(n)
%             
%             if subsp %- if you want to do it on subsamples
%                 for p = 1 : nTau
%                     if length(spikes{n}) > nSpk
%                         %- take a subset of all spike times
%                         rdperm = randperm(length(spikes{n}));
%                         [out{n,p}] = ISI_timescales(sort(spikes{n}(rdperm(1:nSpk))));
%                     else
%                         [out{n,p}] = ISI_timescales([]);
%                         
%                     end
%                     
%                 end
%             else %- to do it on all spikes
%                 [out{n,1}] = ISI_timescales(spikes{n});
%             end
%         end
%         %- save output
%         save([path2go '/processed/ISI_Timescales_' list(d).name(1:end-4) '.mat'],'out','spikes')
%         clear out spikes
%     end
% end
% 
% 


