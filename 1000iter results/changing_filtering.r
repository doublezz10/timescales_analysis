# load data, libraries

library(tidyverse)
library(lmer)
library(lmerTest)
library(ggplot2)
library(scales)
library(emmeans)
library(vroom)

# 0.5 threshold

setwd('/Users/zachz/Documents/timescales_analysis/1000iter results/')

raw_data <- vroom('fixed_single_unit.csv')
raw_data <- raw_data[-c(1)]

data <- raw_data %>% filter(r2 >= 0.5 & between(tau,10,1000) == TRUE)

prop_surviving <- nrow(data)/nrow(raw_data)

grouped_data = data %>% mutate(brain_region = ifelse(brain_area %in% c('acc', 'aca', 'dACC','mcc'), "ACC",
                                               ifelse(brain_area %in% c('amygdala','bla','central'), "Amygdala",
                                                ifelse(brain_area %in% c('hc','ca1','ca2','ca3','dg'), "Hippocampus",
                                                  ifelse(brain_area %in% c('LAI'), "Insula",
                                                    ifelse(brain_area %in% c('mpfc','ila','pl','scACC'), "mPFC",
                                                      ifelse(brain_area %in% c('ofc','orb'), "OFC",
                                                        ifelse(brain_area %in% c('vStriatum','putamen','caudate'), "Striatum",
                                                          "other"))))))))
# graph

se <- function(x) sqrt(var(x)/(length(x)/1000))

grouped_data$brain_region = factor(grouped_data$brain_region,levels=c('Hippocampus','Amygdala','OFC','mPFC','ACC'))

grouped_data %>% 
  filter(brain_region %in% c('OFC','Hippocampus','Amygdala','ACC','mPFC')) %>% group_by(species,brain_region) %>%
  summarise(mean_tau = mean(tau), sd = sd(tau),se = se(tau)) %>%
  ggplot(aes(x=brain_region,y=mean_tau,color=species,fill=species)) +
  geom_point(size=3) + geom_line(aes(group=species))+
  geom_errorbar(aes(ymin=mean_tau-se, ymax=mean_tau+se), width=.1) + 
  xlab('Brain Region') +
  ylab('tau (ms)') + ggtitle('1000 iter, 0.5 cutoff') + ylim(0,550)

# 0.85 threshold

data <- raw_data %>% filter(r2 >= 0.85 & between(tau,10,1000) == TRUE)

prop_surviving <- nrow(data)/nrow(raw_data)

grouped_data = data %>% mutate(brain_region = ifelse(brain_area %in% c('acc', 'aca', 'dACC','mcc'), "ACC",
                                               ifelse(brain_area %in% c('amygdala','bla','central'), "Amygdala",
                                                ifelse(brain_area %in% c('hc','ca1','ca2','ca3','dg'), "Hippocampus",
                                                 ifelse(brain_area %in% c('LAI'), "Insula",
                                                  ifelse(brain_area %in% c('mpfc','ila','pl','scACC'), "mPFC",
                                                   ifelse(brain_area %in% c('ofc','orb'), "OFC",
                                                    ifelse(brain_area %in% c('vStriatum','putamen','caudate'), "Striatum",
                                                     "other"))))))))
# graph

se <- function(x) sqrt(var(x)/(length(x)/1000))

grouped_data$brain_region = factor(grouped_data$brain_region,levels=c('Hippocampus','Amygdala','OFC','mPFC','ACC'))

grouped_data %>% 
  filter(brain_region %in% c('OFC','Hippocampus','Amygdala','ACC','mPFC')) %>% group_by(species,brain_region) %>%
  summarise(mean_tau = mean(tau), sd = sd(tau),se = se(tau)) %>%
  ggplot(aes(x=brain_region,y=mean_tau,color=species,fill=species)) +
  geom_point(size=3) + geom_line(aes(group=species))+
  geom_errorbar(aes(ymin=mean_tau-se, ymax=mean_tau+se), width=.1) + 
  xlab('Brain Region') +
  ylab('tau (ms)') + ggtitle('1000 iter, 0.85 cutoff') + ylim(0,550)

# fred for comparison

df_fred = read.csv('~/Documents/timescales_analysis/fred_results.csv')

df_fred = df_fred %>% mutate(species = factor(species),unitID = as.character(unitID))
df_fred = df_fred %>% rename(unit_id = unitID,
                             brain_area = area,
                             dataset = name,
                             fr = FR)

df_fred_grouped = df_fred %>% mutate(brain_region = ifelse(brain_area %in% c('acc', 'aca', 'dACC'), "ACC",
                                                     ifelse(brain_area %in% c('amygdala','bla','AMG','central'), "Amygdala",
                                                      ifelse(brain_area %in% c('hippocampus','hippocampus2','ca1','ca2','ca3','dg'), "Hippocampus",
                                                       ifelse(brain_area %in% c('LAI'), "Insula",
                                                        ifelse(brain_area %in% c('mpfc','ila','pl','mPFC','scACC'), "mPFC",
                                                         ifelse(brain_area %in% c('OFC','orb','ofc'), "OFC",
                                                          ifelse(brain_area %in% c('ventralStriatum','PUT','Cd'), "Striatum",
                                                           "other"))))))))

df_fred_grouped = df_fred_grouped %>% filter(keep == 1)

se <- function(x) sqrt(var(x)/length(x))

df_fred_grouped$brain_region = factor(df_fred_grouped$brain_region,levels=c('Hippocampus','Amygdala','OFC','mPFC','ACC'))


df_fred_grouped %>% 
  filter(brain_region %in% c('OFC','Hippocampus','Amygdala','ACC','mPFC')) %>% group_by(species,brain_region) %>%
  summarise(mean_tau = mean(tau), sd = sd(tau), se = se(tau)) %>%
  ggplot(aes(x=brain_region,y=mean_tau,color=species,fill=species)) +
  geom_point(size=3) + geom_line(aes(group=species))+
  geom_errorbar(aes(ymin=mean_tau-se, ymax=mean_tau+se), width=.1) + 
  xlab('Brain Region') +
  ylab('tau (ms)') + ggtitle('ISI method') + ylim(0,550)

# 1000 iter pop fit

raw_data = read.csv('fixedpop.csv')

data = raw_data %>% filter(between(tau,10,1000) == TRUE) 

prop_surviving <- nrow(data)/nrow(raw_data)

grouped_data = data %>% mutate(brain_region = ifelse(brain_area %in% c('acc', 'aca', 'dACC','mcc'), "ACC",
                                               ifelse(brain_area %in% c('amygdala','bla','central'), "Amygdala",
                                                ifelse(brain_area %in% c('hc','ca1','ca2','ca3','dg'), "Hippocampus",
                                                 ifelse(brain_area %in% c('LAI'), "Insula",
                                                  ifelse(brain_area %in% c('mpfc','ila','pl','scACC'), "mPFC",
                                                   ifelse(brain_area %in% c('ofc','orb'), "OFC",
                                                    ifelse(brain_area %in% c('vStriatum','putamen','caudate'), "Striatum",
                                                     "other"))))))))

se <- function(x) sqrt(var(x)/(length(x)/1000))

grouped_data$brain_region = factor(grouped_data$brain_region,levels=c('Hippocampus','Amygdala','OFC','mPFC','ACC'))

grouped_data %>% 
  filter(brain_region %in% c('OFC','Hippocampus','Amygdala','ACC','mPFC')) %>% group_by(species,brain_region) %>%
  summarise(mean_tau = mean(tau), sd = sd(tau),se = se(tau)) %>%
  ggplot(aes(x=brain_region,y=mean_tau,color=species,fill=species)) +
  geom_point(size=3) + geom_line(aes(group=species))+
  geom_errorbar(aes(ymin=mean_tau-se, ymax=mean_tau+se), width=.1) + 
  xlab('Brain Region') +
  ylab('tau (ms)') + ggtitle('Population fit 1000x') + ylim(0,550)
