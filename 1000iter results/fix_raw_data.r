library(tidyverse)
library(vroom)

# Load in data

setwd('/Users/zachz/Documents/timescales_analysis/1000iter results/')

raw_data <- vroom('single_units.csv')
raw_data <- raw_data[-c(1)]

fixed_raw <- raw_data %>% mutate(dataset = ifelse(dataset == 'human', 'minxha',dataset))

fixed_raw <- fixed_raw %>% mutate(species = ifelse(species == 'minxha', 'human',species))

fixed_raw <- fixed_raw %>% subset(dataset != 'wirth')

# add in new dataframe with wirth hc and wirth hc2

wirth_data <- vroom('single_units3-2.csv')
wirth_data <- wirth_data[-c(1)]

new_data <- rbind(fixed_raw,wirth_data)

vroom_write(new_data,'fixed_single_unit.csv',delim=',')
