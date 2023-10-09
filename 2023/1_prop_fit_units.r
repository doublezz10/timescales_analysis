# load data

library('tidyverse')
library('vroom')
library('reshape2')

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

yes_survive = filt_br_data %>% filter(survive==1)

ggplot(filt_br_data,aes(x=brain_region)) +
  geom_bar(aes(fill=species, color=species),
           data=filt_br_data,position='dodge',alpha=0.2) +
  geom_bar(aes(fill=species,group=species),data=yes_survive,position='dodge') +
  ggtitle('new data') + ylim(0,4000)
