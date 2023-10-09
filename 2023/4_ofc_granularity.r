# load data

library('tidyverse')
library('vroom')
library('reshape2')
library('RColorBrewer')
library('akima')
library('ggbeeswarm')
library('lme4')
require(gridExtra)

data = vroom('2023_timescales.csv')
data$species = factor(data$species,levels=c('mouse','monkey','human'))

# rename columns, assign brain region

fred_data = data %>% filter(area %in% 
                              c('a11m','a11l','a12m','a12l','a12o','a12r',
                                        'a13m','a13l','LAI'))

fred_data$area = factor(fred_data$area,
                        levels=c('a11m','a11l','a12m','a12l','a12o','a12r',
                                 'a13m','a13l','LAI'))

fred_data = fred_data %>% mutate(granularity = case_when(
  area %in% c('a11m','a11l','a12m','a12l','a12o') ~ 'granular',
  area %in% c('a12r','a13m','a13l') ~ 'dysgranular',
  area %in% c('LAI') ~ 'agranular'
))

fred_data$granularity = factor(fred_data$granularity,
                               levels=c('granular','dysgranular','agranular'))

fred_data$animID = factor(fred_data$animID)

# only keep surviving data

filt_fred_data = fred_data %>% mutate(survive = case_when((
  keep==1 & FR >= 1 & r2 >= 0.5 & tau > 10 & tau < 1000) ~ 1,
  TRUE ~ 0)
)

fred_good_data = filt_fred_data %>% filter(survive==1)

# plot number of units

fred_good_data %>% ggplot(aes(x=area,fill=granularity,color=granularity)) +
  geom_bar(stat="count") + ylab('count neurons')

# timescale by granularity

mean_data = fred_good_data %>% group_by(area,granularity) %>% 
  summarize(mean = mean(tau), se = sd(tau)/sqrt(length((tau))))

mean_data %>% ggplot(aes(x=area,y=mean,group=granularity,fill=granularity)) + 
  geom_errorbar(aes(ymin=mean-se,ymax=mean+se,color=granularity)) + 
  geom_line(aes(color=granularity)) + geom_point(aes(color=granularity)) + 
  ylab('mean timescale (ms)')

# latency by granularity

mean_data2 = fred_good_data %>% group_by(area,granularity) %>% 
  summarize(mean = mean(lat), se = sd(lat)/sqrt(length((lat))))

mean_data2 %>% ggplot(aes(x=area,y=mean,group=granularity,fill=granularity)) + 
  geom_errorbar(aes(ymin=mean-se,ymax=mean+se,color=granularity)) + 
  geom_line(aes(color=granularity)) + geom_point(aes(color=granularity)) + 
  ylab('mean latency (ms)')
  
# xy position (each monkey has different reference, so separate plots by monkey)

monkey1 = fred_good_data %>% filter(animID == 1)
monkey2 = fred_good_data %>% filter(animID == 2)

# average across dv, plot timescale as effect of ap and ml position

mean_space1 = monkey1 %>% group_by(AP,ML) %>% 
  summarize(mean_tau = mean(tau), mean_lat = mean(lat))

mean_space1 %>% ggplot(aes(x=ML,y=AP)) + geom_point(aes(color=mean_tau)) +
  scale_color_viridis_c() 

mean_space2 = monkey2 %>% group_by(AP,ML) %>% 
  summarize(mean_tau = mean(tau), mean_lat = mean(lat))

mean_space2 %>% ggplot(aes(x=ML,y=AP)) + geom_point(aes(color=mean_lat)) + 
 scale_color_viridis_c() 

# spatial interpolation

akima.li <- interp(mean_space1$AP, mean_space1$ML, mean_space1$mean_tau, 
                   xo=seq(min(mean_space1$AP)-1, max(mean_space1$AP)+1, by=1),
                   yo=seq(min(mean_space1$ML)-1, max(mean_space1$ML)+1, by=1),
                   linear=TRUE)

zz = akima.li$z

spatial = data.frame(x=1,y=1,z=1)

for (i in 1:nrow(zz)){
  for (j in 1:ncol(zz)) {
    z_ = zz[i,j]
    col = cbind(i,j,z_)
    spatial[nrow(spatial) + 1, ] <- col 
    
  }
}

spatial1 = spatial[-1,]

monkey1_ = monkey1 %>% rowwise %>% mutate(jitter_ap = AP + (0.02 * sample(-10:10,1)), 
                                          jitter_ml = ML + (0.02 * sample(-10:10,1)))

spatial1 %>% ggplot() + geom_tile(aes(x=y*-1,y=x,fill=z)) + 
  scale_fill_viridis_c(limits=(c(0,1000))) +
  ylab('A/P position (mm)') + xlab('M/L position (mm)') + 
  labs(fill = "timescale (ms)") + ggtitle('Monkey 1') +
  geom_point(data=monkey1_,aes(x=-1*jitter_ml-24,y=jitter_ap-31.25),size=0.05)

# keep working on jitter, smaller points

# repeat for monkey 2

akima.li <- interp(mean_space2$AP, mean_space2$ML, mean_space2$mean_tau, 
                   xo=seq(min(mean_space2$AP)- 1, max(mean_space2$AP)+ 1, by=1),
                   yo=seq(min(mean_space2$ML)- 1, max(mean_space2$ML)+ 1, by=1),
                   linear=TRUE)

zz = akima.li$z

spatial = data.frame(x=1,y=1,z=1)

for (i in 1:nrow(zz)){
  for (j in 1:ncol(zz)) {
    z_ = zz[i,j]
    col = cbind(i,j,z_)
    spatial[nrow(spatial) + 1, ] <- col 
    
  }
}

spatial2 = spatial[-1,]

monkey2_ = monkey2 %>% rowwise %>% mutate(jitter_ap = AP + (0.02 * sample(-10:10,1)), 
                    jitter_ml = ML + (0.02 * sample(-10:10,1)))

spatial2 %>% ggplot() + geom_tile(aes(x=y,y=x,fill=z)) + 
  scale_fill_viridis_c(limits=(c(0,1000))) +
  ylab('A/P position (mm)') + xlab('M/L position (mm)') + 
  labs(fill = "timescale (ms)") + ggtitle('Monkey 2') +
  geom_point(data=monkey2_,aes(x=jitter_ml-3,y=jitter_ap-29.25),size=0.05)


model1 = lm(data=mean_space1,'mean_tau ~ AP * ML')
summary(model1)

model2 = lm(data=mean_space2,'mean_tau ~ AP * ML')
summary(model2)

# do stats within each area 

areas = list('a11m','a11l','a12m','a12l','a12o','a12r',
               'a13m','a13l','LAI')

pvals = data.frame()

for (this_area in areas) {
  model1 = lm('tau ~ AP * ML',monkey1 %>% filter(area==this_area))
  
  print(this_area)
  
  model2 = lm('tau ~ AP * ML',monkey2 %>% filter(area==this_area))
  
  ps = c(this_area,summary(model1)$coefficients[,4][2:4],summary(model2)$coefficients[,4][2:4])
  
  pvals = rbind(pvals,ps)
}

colnames(pvals) = c('area','AP_1','ML_1','interaction_1','AP_2','ML_2','interaction_2')

# make pvalues actual numbers :)
i = c(2,3,4,5,6,7)
pvals[ , i] <- apply(pvals[ , i], 2,function(x) as.numeric(as.character(x)))

pvals2 = pvals %>% pivot_longer(cols=!area,
                       names_to=c('direction','animal'),names_pattern='(.*)_(.)')

pvals2 = pvals2 %>% mutate(adj_pval = p.adjust(pvals2$value))
pvals2$direction = factor(pvals2$direction,levels=c('interaction','ML','AP'))

plot1 = pvals2 %>% filter(animal==1) %>% 
  ggplot(aes(x=area,y=direction)) + geom_tile(aes(fill=value)) + 
  ggtitle('Monkey 1') + scale_fill_fermenter(breaks=c(0.01,0.05,0.10,0.5))

plot2 = pvals2 %>% filter(animal==2) %>% 
  ggplot(aes(x=area,y=direction)) + geom_tile(aes(fill=value)) +
  ggtitle('Monkey 2') + scale_fill_fermenter(breaks=c(0.01,0.05,0.10,0.5))

grid.arrange(plot1, plot2, nrow=2)

# # are there gradients within groups of areas (11m + l, etc); not working quite yet

# group1 = c('11m','11l')
# group2 = c('12m','12l','12r','12o')
# group3 = c('13m','13l')
# groups = list(group1,group2,group3)
# 
# pvals = data.frame()
# 
# for (group in groups) {
#   model1 = lm('tau ~ AP * ML',monkey1 %>% filter(area %in% group))
#   
#   print(this_area)
#   
#   model2 = lm('tau ~ AP * ML',monkey2 %>% filter(area %in% group))
#   
#   ps = c(group[1][1:2],summary(model1)$coefficients[,4][2:4],summary(model2)$coefficients[,4][2:4])
#   
#   pvals = rbind(pvals,ps)
# }
# 
# colnames(pvals) = c('area','AP_1','ML_1','interaction_1','AP_2','ML_2','interaction_2')
# 
# # make pvalues actual numbers :)
# i = c(2,3,4,5,6,7)
# pvals[ , i] <- apply(pvals[ , i], 2,function(x) as.numeric(as.character(x)))
# 
# pvals2 = pvals %>% pivot_longer(cols=!area,
#                                 names_to=c('direction','animal'),names_pattern='(.*)_(.)')
# 
# pvals2 = pvals2 %>% mutate(adj_pval = p.adjust(pvals2$value))
# pvals2$direction = factor(pvals2$direction,levels=c('interaction','ML','AP'))
# 
# plot1 = pvals2 %>% filter(animal==1) %>% 
#   ggplot(aes(x=area,y=direction)) + geom_tile(aes(fill=value)) + 
#   ggtitle('Morbier') + scale_fill_fermenter(breaks=c(0.01,0.05,0.10,0.5))
# 
# plot2 = pvals2 %>% filter(animal==2) %>% 
#   ggplot(aes(x=area,y=direction)) + geom_tile(aes(fill=value)) +
#   ggtitle('Mimic') + scale_fill_fermenter(breaks=c(0.01,0.05,0.10,0.5))
# 
# grid.arrange(plot1, plot2, nrow=2)
# 
