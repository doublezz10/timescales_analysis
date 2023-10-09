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

old_data = vroom('old_data.csv')

# loop through and find matching values

