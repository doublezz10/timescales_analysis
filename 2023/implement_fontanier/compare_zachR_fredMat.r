library(tidyverse)
library('vroom')
library(lme4)

# fontanier mcc

zach = vroom('font_mcc_results.csv')
fred = vroom('2023_timescales.csv')

zach$unitID = zach$unit

fred_font = fred %>% filter(name=='fontanier' & keep==1)

combined = inner_join(zach,fred_font,by='unitID')

# x's are zach's implementation in R; y's are Fred's in MATLAB

plot(combined$tau.x,combined$tau.y)

model = lm('tau.y ~ tau.x',combined)
summary(model)

plot(combined$lat.x,combined$lat.y)

model = lm('lat.y ~ lat.x',combined)
summary(model)

plot(combined$fr,combined$FR)

model = lm('FR ~ fr',combined)
summary(model)

# repeat for stoll amg