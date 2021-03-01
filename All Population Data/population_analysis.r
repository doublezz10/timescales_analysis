library(tidyverse)
library(lme4)
library(lmerTest)
library(ggplot2)
library(ggpubr)
library(scales)
library(emmeans)

## Import data

# Zach's regular method

df_pop = read.csv('~/Documents/timescales_analysis/All Population Data/all_population_data.csv')

## Plot population values for each species

# Mouse

mouse_tau = df_pop %>% filter(dataset == 'steinmetz') %>% 
  mutate(brain_area = fct_reorder(brain_area, tau, .fun='mean')) %>% 
  ggplot(aes(x=brain_area,y=tau,fill=fr)) +
  geom_point() + ggtitle('Mouse')

mouse_n = df_pop %>% filter(dataset == 'steinmetz') %>% 
  mutate(brain_area = fct_reorder(brain_area, tau, .fun='mean')) %>% 
  ggplot(aes(x=brain_area,y=n,fill=dataset)) +
  geom_col(position='stack') +  ggtitle('Mouse')

mouse_fig = ggarrange(mouse_tau,mouse_n,labels=c('a','b'),ncol=2,nrow=1)

mouse_fig

# Rat

rat_tau = df_pop %>% filter(dataset == 'buzsaki' | dataset == 'lemerre' | dataset == 'peyrache') %>% 
  mutate(brain_area = fct_reorder(brain_area, tau, .fun='mean')) %>% 
  ggplot(aes(x=reorder(brain_area,tau),y=tau,fill=fr)) +
  geom_point(position='dodge') + ggtitle('Rat')

rat_n = df_pop %>% filter(dataset == 'buzsaki' | dataset == 'lemerre' | dataset == 'peyrache') %>% 
  mutate(brain_area = fct_reorder(brain_area, tau, .fun='mean')) %>% 
  ggplot(aes(x=brain_area,y=n,fill=dataset)) +
  geom_col(position='stack') +  ggtitle('Rat')

rat_fig = ggarrange(rat_tau,rat_n,labels=c('a','b'),ncol=2,nrow=1)

rat_fig

# Monkey

monkey_tau = df_pop %>% filter(dataset == 'stoll' | dataset == 'meg' | dataset == 'wirth' | dataset == 'froot') %>% 
  mutate(brain_area = fct_reorder(brain_area, tau, .fun='mean')) %>% 
  ggplot(aes(x=brain_area,y=tau,fill=fr)) +
  geom_point() + ggtitle('Monkey')

monkey_n = df_pop %>% filter(dataset == 'stoll' | dataset == 'meg' | dataset == 'wirth' | dataset == 'froot') %>% 
  mutate(brain_area = fct_reorder(brain_area, tau, .fun='mean')) %>% 
  ggplot(aes(x=brain_area,y=n,fill=dataset)) +
  geom_col(position='stack') +  ggtitle('Monkey')

monkey_fig = ggarrange(monkey_tau,monkey_n,labels=c('a','b'),ncol=2,nrow=1)

monkey_fig

# Human

human_tau = df_pop %>% filter(dataset == 'faraut' | dataset == 'minxha') %>% 
  mutate(brain_area = fct_reorder(brain_area, tau, .fun='mean')) %>% 
  ggplot(aes(x=brain_area,y=tau,fill=fr)) +
  geom_point() + ggtitle('Human')

human_n = df_pop %>% filter(dataset == 'faraut' | dataset == 'minxha') %>% 
  mutate(brain_area = fct_reorder(brain_area, tau, .fun='mean')) %>% 
  ggplot(aes(x=brain_area,y=n,fill=dataset)) +
  geom_col(position='stack') +  ggtitle('Human')

human_fig = ggarrange(human_tau,human_n,labels=c('a','b'),ncol=2,nrow=1)

human_fig

## Repeat on Fred's data

df_fred = read.csv('~/Documents/timescales_analysis/GLM/fred_data_v3.csv')

df_fred = df_fred %>% mutate(species = factor(species),unitID = as.character(unitID))
df_fred = df_fred %>% rename(unit_id = unitID,
                             brain_area = area,
                             dataset = name,
                             fr = FR)

fred_means_df = df_fred %>% group_by(species,brain_area) %>% summarize(mean_tau = mean(tau))

fred_means_df %>% filter(species == 'mouse') %>% mutate(brain_area = fct_reorder(brain_area, mean_tau, .fun='mean')) %>%
  ggplot(aes(x=brain_area,y=mean_tau)) + geom_point() + ggtitle('Mouse')

fred_means_df %>% filter(species == 'rat') %>% mutate(brain_area = fct_reorder(brain_area, mean_tau, .fun='mean')) %>%
  ggplot(aes(x=brain_area,y=mean_tau)) + geom_point() + ggtitle('Rat')

fred_means_df %>% filter(species == 'monkey') %>% mutate(brain_area = fct_reorder(brain_area, mean_tau, .fun='mean')) %>%
  ggplot(aes(x=brain_area,y=mean_tau)) + geom_point() + ggtitle('Monkey')

fred_means_df %>% filter(species == 'human') %>% mutate(brain_area = fct_reorder(brain_area, mean_tau, .fun='mean')) %>%
  ggplot(aes(x=brain_area,y=mean_tau)) + geom_point() + ggtitle('Human')

## One summary figure with only brain regions in common

# Zach

grouped_df_pop = df_pop  %>% mutate(brain_region = ifelse(brain_area %in% c('acc', 'aca', 'dacc', 'scacc'), "ACC",
                                  ifelse(brain_area %in% c('amyg','bla','central'), "Amygdala",
                                  ifelse(brain_area %in% c('hc','ca1','ca2','ca3','dg'), "Hippocampus",
                                  ifelse(brain_area %in% c('LAI'), "Insula",
                                  ifelse(brain_area %in% c('mpfc','ila','pl','mpfc'), "mPFC",
                                  ifelse(brain_area %in% c('ofc','orb','ofc'), "OFC",
                                  ifelse(brain_area %in% c('vs','put','cd'), "Striatum",
                                     "other"))))))))

grouped_df_pop = grouped_df_pop %>% mutate(species = ifelse(dataset %in% c('steinmetz'),'mouse',
                                                     ifelse(dataset %in% c('peyrache','buzsaki','lemerre'),'rat',
                                                     ifelse(dataset %in% c('stoll','meg','wirth','froot'),'monkey',
                                                     ifelse(dataset %in% c('faraut','minxha'),'human',
                                                            '?')))))

grouped_df_pop %>% filter(brain_region != 'other' | brain_region != 'striatum') %>%
  ggplot(aes(x=brain_region,y=tau,fill=species,color=species)) + geom_point()
