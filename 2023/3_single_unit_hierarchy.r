# load data

library('tidyverse')
library('vroom')
library('reshape2')
library('lme4')
library('emmeans')

data = vroom('2023_timescales.csv')
data$species = factor(data$species,levels=c('mouse','monkey','human'))

# rename columns, assign brain region

data = data %>% mutate(brain_region = case_when(
  area %in% c('amyg','amygdala','bla','AMG') ~ 'amygdala',
  area %in% c('hc','hippocampus','ca1','ca2','ca3','dg','hippocampus2') ~ 'hippocampus',
  area %in% c('mcc','dACC','aca','ACC') ~ 'ACC',
  area %in% c('scACC','ila','pl') ~ 'mPFC',
  area %in% c('OFC','orb','a11l','a11m','a13l','13m') ~ 'OFC'
))

# filter data

br_data = data %>% filter(is.na(brain_region)==FALSE)
br_data$brain_region = factor(br_data$brain_region,
                              levels=c('hippocampus','amygdala','OFC','mPFC','ACC'))

filt_br_data = br_data %>% mutate(survive = case_when((
  keep==1 & FR >= 1 & r2 >= 0.5 & tau > 10 & tau < 1000) ~ 1,
  TRUE ~ 0)
)

# only keep surviving data

good_data = filt_br_data %>% filter(survive==1)

# summarize and plot

mean_data = good_data %>% group_by(species,brain_region) %>% 
  summarize(mean = mean(tau), se = sd(tau)/sqrt(length((tau))))

mean_data %>% ggplot(aes(x=brain_region,y=mean,group=species,fill=species)) +
  geom_ribbon(aes(ymin=mean-se,ymax=mean+se),alpha=0.3) + 
  geom_line(aes(color=species)) + geom_point(aes(color=species)) +
  ylab('mean timescale (ms)')
  
# repeat for latency
  
mean_data2 = good_data %>% group_by(species,brain_region) %>% 
  summarize(mean = mean(lat), se = sd(lat)/sqrt(length((lat))))
  
  mean_data2 %>% ggplot(aes(x=brain_region,y=mean,group=species,fill=species)) +
    geom_ribbon(aes(ymin=mean-se,ymax=mean+se),alpha=0.3) + 
    geom_line(aes(color=species)) + geom_point(aes(color=species)) +
    ylab('mean latency (ms)')
  
# models

tau_model = lm('tau ~ brain_region + species',good_data)
summary(tau_model)

emm1 = emmeans(tau_model, specs = pairwise ~ species|brain_region)
emm1$contrasts

pwpp(emm1)

lat_model = lm('lat ~ brain_region + species',good_data)
summary(tau_model)

emm2 = emmeans(lat_model, specs = pairwise ~ species|brain_region)
emm2$contrasts

pwpp(emm2)