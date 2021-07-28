library(tidyverse)
library(lme4)
library(lmerTest)
library(ggplot2)
library(scales)
library(emmeans)

emm_options(pbkrtest.limit = 4000)

## Pretty plot fxns

GeomSplitViolin <- ggproto("GeomSplitViolin", GeomViolin, 
                           draw_group = function(self, data, ..., draw_quantiles = NULL) {
                             data <- transform(data, xminv = x - violinwidth * (x - xmin), xmaxv = x + violinwidth * (xmax - x))
                             grp <- data[1, "group"]
                             newdata <- plyr::arrange(transform(data, x = if (grp %% 2 == 1) xminv else xmaxv), if (grp %% 2 == 1) y else -y)
                             newdata <- rbind(newdata[1, ], newdata, newdata[nrow(newdata), ], newdata[1, ])
                             newdata[c(1, nrow(newdata) - 1, nrow(newdata)), "x"] <- round(newdata[1, "x"])
                             
                             if (length(draw_quantiles) > 0 & !scales::zero_range(range(data$y))) {
                               stopifnot(all(draw_quantiles >= 0), all(draw_quantiles <=
                                                                         1))
                               quantiles <- ggplot2:::create_quantile_segment_frame(data, draw_quantiles)
                               aesthetics <- data[rep(1, nrow(quantiles)), setdiff(names(data), c("x", "y")), drop = FALSE]
                               aesthetics$alpha <- rep(1, nrow(quantiles))
                               both <- cbind(quantiles, aesthetics)
                               quantile_grob <- GeomPath$draw_panel(both, ...)
                               ggplot2:::ggname("geom_split_violin", grid::grobTree(GeomPolygon$draw_panel(newdata, ...), quantile_grob))
                             }
                             else {
                               ggplot2:::ggname("geom_split_violin", GeomPolygon$draw_panel(newdata, ...))
                             }
                           })

geom_split_violin <- function(mapping = NULL, data = NULL, stat = "ydensity", position = "identity", ..., 
                              draw_quantiles = NULL, trim = TRUE, scale = "area", na.rm = FALSE, 
                              show.legend = NA, inherit.aes = TRUE) {
  layer(data = data, mapping = mapping, stat = stat, geom = GeomSplitViolin, 
        position = position, show.legend = show.legend, inherit.aes = inherit.aes, 
        params = list(trim = trim, scale = scale, draw_quantiles = draw_quantiles, na.rm = na.rm, ...))
}

"%||%" <- function(a, b) {
  if (!is.null(a)) a else b
}

geom_flat_violin <- function(mapping = NULL, data = NULL, stat = "ydensity",
                             position = "dodge", trim = TRUE, scale = "area",
                             show.legend = NA, inherit.aes = TRUE, ...) {
  layer(
    data = data,
    mapping = mapping,
    stat = stat,
    geom = GeomFlatViolin,
    position = position,
    show.legend = show.legend,
    inherit.aes = inherit.aes,
    params = list(
      trim = trim,
      scale = scale,
      ...
    )
  )
}

#' @rdname ggplot2-ggproto
#' @format NULL
#' @usage NULL
#' @export
GeomFlatViolin <-
  ggproto("GeomFlatViolin", Geom,
          setup_data = function(data, params) {
            data$width <- data$width %||%
              params$width %||% (resolution(data$x, FALSE) * 0.9)
            
            # ymin, ymax, xmin, and xmax define the bounding rectangle for each group
            data %>%
              group_by(group) %>%
              mutate(
                ymin = min(y),
                ymax = max(y),
                xmin = x,
                xmax = x + width / 2
              )
          },
          
          draw_group = function(data, panel_scales, coord) {
            # Find the points for the line to go all the way around
            data <- transform(data,
                              xminv = x,
                              xmaxv = x + violinwidth * (xmax - x)
            )
            
            # Make sure it's sorted properly to draw the outline
            newdata <- rbind(
              plyr::arrange(transform(data, x = xminv), y),
              plyr::arrange(transform(data, x = xmaxv), -y)
            )
            
            # Close the polygon: set first and last point the same
            # Needed for coord_polar and such
            newdata <- rbind(newdata, newdata[1, ])
            
            ggplot2:::ggname("geom_flat_violin", GeomPolygon$draw_panel(newdata, panel_scales, coord))
          },
          
          draw_key = draw_key_polygon,
          
          default_aes = aes(
            weight = 1, colour = "grey20", fill = "white", size = 0.5,
            alpha = NA, linetype = "solid"
          ),
          
          required_aes = c("x", "y")
  )

## Import Fred's data

df_fred = read.csv('~/Documents/timescales_analysis/fred_results.csv')

df_fred = df_fred %>% mutate(species = factor(species),unitID = as.character(unitID))
df_fred = df_fred %>% rename(unit_id = unitID,
                             brain_area = area,
                             dataset = name,
                             fr = FR)

## Add new column to group by brain region (i.e. collapse subregions)

df_fred_grouped = df_fred %>% mutate(brain_region = ifelse(brain_area %in% c('acc', 'aca', 'dACC'), "ACC",
                                                        ifelse(brain_area %in% c('amygdala','bla','AMG','central'), "Amygdala",
                                                        ifelse(brain_area %in% c('hippocampus','hippocampus2','ca1','ca2','ca3','dg'), "Hippocampus",
                                                        ifelse(brain_area %in% c('LAI'), "Insula",
                                                        ifelse(brain_area %in% c('mpfc','ila','pl','mPFC','scACC'), "mPFC",
                                                        ifelse(brain_area %in% c('OFC','orb','ofc'), "OFC",
                                                        ifelse(brain_area %in% c('ventralStriatum','PUT','Cd'), "Striatum",
                                                          "other"))))))))

df_fred_grouped = df_fred_grouped %>% filter(keep == 1)

df_fred_grouped %>% filter(brain_region != 'other') %>% ggplot(aes(x=species,y=tau,color=brain_region,fill=brain_region)) +
  geom_flat_violin(position=position_nudge(x=0.2,y=0),adjust=2,alpha=0.5) + 
  geom_point(position=position_jitter(width=0.15),size=0.25) + coord_flip() +
  ggtitle('All Units') +theme(plot.title = element_text(hjust = 0.5),legend.position='bottom') +
  scale_y_log10(breaks = scales::trans_breaks("log10", function(x) 10^x),
                labels = scales::trans_format("log10", scales::math_format(10^.x)),
                limits = c(9,1100)) +
  scale_x_discrete(limits = c("mouse", "rat", "monkey","human"))

lmer(tau ~ species * brain_region + (1|fr), 
     data = df_fred_grouped %>% filter(brain_region != 'other')) %>% summary()

## Separate by brain region, plot, GLMs

acc_fred = df_fred_grouped %>% filter(brain_region == 'ACC')
amyg_fred = df_fred_grouped %>% filter(brain_region == 'Amygdala')
hc_fred = df_fred_grouped %>% filter(brain_region == 'Hippocampus')
insula_fred = df_fred_grouped %>% filter(brain_region == 'Insula')
mpfc_fred = df_fred_grouped %>% filter(brain_region == 'mPFC')
ofc_fred = df_fred_grouped %>% filter(brain_region == 'OFC')
str_fred = df_fred_grouped %>% filter(brain_region == 'Striatum')

# ACC

acc_fred %>% ggplot(aes(x=species,y=tau,color=brain_area,fill=brain_area)) +
  geom_flat_violin(position=position_nudge(x=0.2,y=0),adjust=2,alpha=0.5) + 
  geom_point(position=position_jitter(width=0.15),size=0.25) + coord_flip() +
  ggtitle('ACC') +theme(plot.title = element_text(hjust = 0.5),legend.position='bottom') +
  scale_y_log10(breaks = scales::trans_breaks("log10", function(x) 10^x),
                labels = scales::trans_format("log10", scales::math_format(10^.x)),
                limits = c(9,1100)) +
  scale_x_discrete(limits = c("mouse", "rat", "monkey","human"))

lmer(tau ~ species + (1|fr), data = acc_fred) %>% summary()
acc_fred_emm = lmer(tau ~ species + (1|fr), data = acc_fred) %>% emmeans(~ species)
pwpp(acc_fred_emm, comparisions = TRUE) + ggtitle('ACC')

lmer(tau ~ species + brain_area + (1|fr), data = acc_fred) %>% summary()

# Amygdala

amyg_fred %>% ggplot(aes(x=species,y=tau,color=brain_area,fill=brain_area)) +
  geom_flat_violin(position=position_nudge(x=0.2,y=0),adjust=2,alpha=0.5) + 
  geom_point(position=position_jitter(width=0.15),size=0.25) + coord_flip() +
  ggtitle('Amygdala') +theme(plot.title = element_text(hjust = 0.5),legend.position='bottom') +
  scale_y_log10(breaks = scales::trans_breaks("log10", function(x) 10^x),
                labels = scales::trans_format("log10", scales::math_format(10^.x)),
                limits = c(9,1100)) +
  scale_x_discrete(limits = c("mouse", "rat", "monkey","human"))

amyg_with_fr = lm(tau ~ species + fr, data = amyg_fred)
amyg_no_fr = lm(tau~species, data = amyg_fred)

summary(amyg_with_fr)
summary(amyg_no_fr)

anova(amyg_no_fr,amyg_with_fr)

# Hippocampus

hc_fred %>% ggplot(aes(x=species,y=tau,color=brain_area,fill=brain_area)) +
  geom_flat_violin(position=position_nudge(x=0.2,y=0),adjust=2,alpha=0.5) + 
  geom_point(position=position_jitter(width=0.15),size=0.25) + coord_flip() +
  ggtitle('Hippocampus') +theme(plot.title = element_text(hjust = 0.5),legend.position='bottom') +
  scale_y_log10(breaks = scales::trans_breaks("log10", function(x) 10^x),
                labels = scales::trans_format("log10", scales::math_format(10^.x)),
                limits = c(9,1100)) +
  scale_x_discrete(limits = c("mouse", "rat", "monkey","human"))

lmer(tau ~ species + (1|fr), data = hc_fred) %>% summary()
hc_fred_emm = lmer(tau ~ species + (1|fr), data = hc_fred) %>% emmeans(~ species)
pwpp(hc_fred_emm, comparisons = TRUE) + ggtitle('Hippocampus')

lmer(tau ~ species + brain_area + (1|fr), data = hc_fred) %>% summary()

# Insula

insula_fred %>% ggplot(aes(x=species,y=tau,color=brain_area,fill=brain_area)) +
  geom_flat_violin(position=position_nudge(x=0.2,y=0),adjust=2,alpha=0.5) + 
  geom_point(position=position_jitter(width=0.15),size=0.25) + coord_flip() +
  ggtitle('Insula') +theme(plot.title = element_text(hjust = 0.5),legend.position='bottom') +
  scale_y_log10(breaks = scales::trans_breaks("log10", function(x) 10^x),
                labels = scales::trans_format("log10", scales::math_format(10^.x)),
                limits = c(9,1100)) +
  scale_x_discrete(limits = c("mouse", "rat", "monkey","human"))

# mPFC

mpfc_fred %>% ggplot(aes(x=species,y=tau,color=brain_area,fill=brain_area)) +
  geom_flat_violin(position=position_nudge(x=0.2,y=0),adjust=2,alpha=0.5) + 
  geom_point(position=position_jitter(width=0.15),size=0.25) + coord_flip() +
  ggtitle('mPFC') +theme(plot.title = element_text(hjust = 0.5),legend.position='bottom') +
  scale_y_log10(breaks = scales::trans_breaks("log10", function(x) 10^x),
                labels = scales::trans_format("log10", scales::math_format(10^.x)),
                limits = c(9,1100)) +
  scale_x_discrete(limits = c("mouse", "rat", "monkey","human"))

lmer(tau ~ species + (1|fr), data = mpfc_fred) %>% summary()
mpfc_fred_emm = lmer(tau ~ species + (1|fr), data = mpfc_fred) %>% emmeans(~ species)
pwpp(mpfc_fred_emm, comparisons = TRUE) + ggtitle('mPFC')

lmer(tau ~ species + brain_area + (1|fr), data = mpfc_fred) %>% summary()
lmer(tau ~ species + brain_area + (1|fr), data = mpfc_fred) %>% emmeans(pairwise ~ species * brain_area)

# OFC

ofc_fred %>% ggplot(aes(x=species,y=tau,color=brain_area,fill=brain_area)) +
  geom_flat_violin(position=position_nudge(x=0.2,y=0),adjust=2,alpha=0.5) + 
  geom_point(position=position_jitter(width=0.15),size=0.25) + coord_flip() +
  ggtitle('OFC') +theme(plot.title = element_text(hjust = 0.5),legend.position='bottom') +
  scale_y_log10(breaks = scales::trans_breaks("log10", function(x) 10^x),
                labels = scales::trans_format("log10", scales::math_format(10^.x)),
                limits = c(9,1100)) +
  scale_x_discrete(limits = c("mouse", "rat", "monkey","human"))

lmer(tau ~ species + (1|fr), data = ofc_fred) %>% summary()
ofc_fred_emm = lmer(tau ~ species + (1|fr), data = ofc_fred) %>% emmeans(~ species)
pwpp(ofc_fred_emm, comparisons = TRUE) + ggtitle('OFC')

lmer(tau ~ species + brain_area + (1|fr), data = ofc_fred) %>% summary()
lmer(tau ~ species + brain_area + (1|fr), data = ofc_fred) %>% emmeans(pairwise ~ species * brain_area)

# Striatum

str_fred %>% ggplot(aes(x=species,y=tau,color=brain_area,fill=brain_area)) +
  geom_flat_violin(position=position_nudge(x=0.2,y=0),adjust=2,alpha=0.5) + 
  geom_point(position=position_jitter(width=0.15),size=0.25) + coord_flip() +
  ggtitle('Striatum') +theme(plot.title = element_text(hjust = 0.5),legend.position='bottom') +
  scale_y_log10(breaks = scales::trans_breaks("log10", function(x) 10^x),
                labels = scales::trans_format("log10", scales::math_format(10^.x)),
                limits = c(9,1100)) +
  scale_x_discrete(limits = c("mouse", "rat", "monkey","human"))

## OFC vs Insula

ofc_ins_fred = df_fred_grouped %>% filter(brain_region == 'OFC' | brain_region == 'Insula')

ofc_ins_fred %>% ggplot(aes(x=species,y=tau,color=brain_area,fill=brain_area)) +
  geom_flat_violin(position=position_nudge(x=0.2,y=0),adjust=2,alpha=0.5) + 
  geom_point(position=position_jitter(width=0.15),size=0.25) + coord_flip() +
  ggtitle('OFC & Insula') +theme(plot.title = element_text(hjust = 0.5),legend.position='bottom') +
  scale_y_log10(breaks = scales::trans_breaks("log10", function(x) 10^x),
                labels = scales::trans_format("log10", scales::math_format(10^.x)),
                limits = c(9,1100)) +
  scale_x_discrete(limits = c("mouse", "rat", "monkey","human"))

## Within species just for fun

df_fred_grouped %>% filter(species == 'mouse') %>% ggplot(aes(x=brain_region,y=tau,color=dataset,fill=dataset)) +
  geom_flat_violin(position=position_nudge(x=0.2,y=0),adjust=2,alpha=0.5) + 
  geom_point(position=position_jitter(width=0.15),size=0.25) + coord_flip() +
  ggtitle('Mouse') +theme(plot.title = element_text(hjust = 0.5),legend.position='bottom') +
  scale_y_log10(breaks = scales::trans_breaks("log10", function(x) 10^x),
                labels = scales::trans_format("log10", scales::math_format(10^.x)),
                limits = c(9,1100))

df_fred_grouped %>% filter(species == 'rat') %>% ggplot(aes(x=brain_region,y=tau,color=dataset,fill=dataset)) +
  geom_flat_violin(position=position_nudge(x=0.2,y=0),adjust=2,alpha=0.5) + 
  geom_point(position=position_jitter(width=0.15),size=0.25) + coord_flip() +
  ggtitle('Rat') +theme(plot.title = element_text(hjust = 0.5),legend.position='bottom') +
  scale_y_log10(breaks = scales::trans_breaks("log10", function(x) 10^x),
                labels = scales::trans_format("log10", scales::math_format(10^.x)),
                limits = c(9,1100))

df_fred_grouped %>% filter(species == 'monkey') %>% ggplot(aes(x=brain_region,y=tau,color=dataset,fill=dataset)) +
  geom_flat_violin(position=position_nudge(x=0.2,y=0),adjust=2,alpha=0.5) + 
  geom_point(position=position_jitter(width=0.15),size=0.25) + coord_flip() +
  ggtitle('Monkey') +theme(plot.title = element_text(hjust = 0.5),legend.position='bottom') +
  scale_y_log10(breaks = scales::trans_breaks("log10", function(x) 10^x),
                labels = scales::trans_format("log10", scales::math_format(10^.x)),
                limits = c(9,1100))

df_fred_grouped %>% filter(species == 'human') %>% ggplot(aes(x=brain_region,y=tau,color=dataset,fill=dataset)) +
  geom_flat_violin(position=position_nudge(x=0.2,y=0),adjust=2,alpha=0.5) + 
  geom_point(position=position_jitter(width=0.15),size=0.25) + coord_flip() +
  ggtitle('Human') +theme(plot.title = element_text(hjust = 0.5),legend.position='bottom') +
  scale_y_log10(breaks = scales::trans_breaks("log10", function(x) 10^x),
                labels = scales::trans_format("log10", scales::math_format(10^.x)),
                limits = c(9,1100))

## Nice line graph

se <- function(x) sqrt(var(x)/length(x))

df_fred_grouped$brain_region = factor(df_fred_grouped$brain_region,levels=c('Hippocampus','Amygdala','OFC','mPFC','ACC'))


df_fred_grouped %>% 
  filter(brain_region %in% c('OFC','Hippocampus','Amygdala','ACC','mPFC')) %>% group_by(species,brain_region) %>%
  summarise(mean_tau = mean(tau), sd = sd(tau), se = se(tau)) %>%
  ggplot(aes(x=brain_region,y=mean_tau,color=species,fill=species)) +
  geom_point(size=3) + geom_line(aes(group=species))+
  geom_errorbar(aes(ymin=mean_tau-se, ymax=mean_tau+se), width=.1) + 
  xlab('Brain Region') +
  ylab('tau (ms)') + ggtitle('ISI method')

