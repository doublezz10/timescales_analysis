# load data, libraries

library(tidyverse)
library(lmer)
library(lmerTest)
library(ggplot2)
library(scales)
library(emmeans)

setwd('/Users/zachz/Documents/timescales_analysis/1000iter results/')

raw_data = read.csv('fixedpop.csv')

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


# filter, plot, lm

data = raw_data %>% filter(between(tau,10,1000) == TRUE) 

prop_surviving <- nrow(data)/nrow(raw_data)

grouped_data = data %>% mutate(brain_region = ifelse(brain_area %in% c('acc', 'aca', 'dACC'), "ACC",
                                              ifelse(brain_area %in% c('amygdala','bla','central'), "Amygdala",
                                              ifelse(brain_area %in% c('hc','ca1','ca2','ca3','dg'), "Hippocampus",
                                              ifelse(brain_area %in% c('LAI'), "Insula",
                                              ifelse(brain_area %in% c('mpfc','ila','pl','scACC'), "mPFC",
                                              ifelse(brain_area %in% c('ofc','orb'), "OFC",
                                              ifelse(brain_area %in% c('vStriatum','putamen','caudate'), "Striatum",
                                              "other"))))))))

grouped_data %>% filter(brain_region != 'other') %>% ggplot(aes(x=species,y=tau,color=brain_region,fill=brain_region)) +
  geom_flat_violin(position=position_nudge(x=0.2,y=0),adjust=2,alpha=0.5) + 
  geom_point(position=position_jitter(width=0.15),size=0.25) + coord_flip() +
  ggtitle('All Units') +theme(plot.title = element_text(hjust = 0.5),legend.position='bottom') +
  scale_y_log10(breaks = scales::trans_breaks("log10", function(x) 10^x),
                labels = scales::trans_format("log10", scales::math_format(10^.x)),
                limits = c(9,1100)) +
  scale_x_discrete(limits = c("mouse", "rat", "monkey","human"))

lm(tau ~ species * brain_region + fr, 
     data = grouped_data %>% filter(brain_region != 'other')) %>% summary()

# split by brain_region, then repeat

## Separate by brain region, plot, GLMs

acc = grouped_data %>% filter(brain_region == 'ACC')
amyg = grouped_data %>% filter(brain_region == 'Amygdala')
hc = grouped_data %>% filter(brain_region == 'Hippocampus')
insula = grouped_data %>% filter(brain_region == 'Insula')
mpfc = grouped_data %>% filter(brain_region == 'mPFC')
ofc = grouped_data %>% filter(brain_region == 'OFC')
str = grouped_data %>% filter(brain_region == 'Striatum')

# ACC

acc %>% ggplot(aes(x=species,y=tau,color=brain_area,fill=brain_area)) +
  geom_flat_violin(position=position_nudge(x=0.2,y=0),adjust=2,alpha=0.5) + 
  geom_point(position=position_jitter(width=0.15),size=0.25) + coord_flip() +
  ggtitle('ACC - pop fit') +theme(plot.title = element_text(hjust = 0.5),legend.position='bottom') +
  scale_y_log10(breaks = scales::trans_breaks("log10", function(x) 10^x),
                labels = scales::trans_format("log10", scales::math_format(10^.x)),
                limits = c(9,1100)) +
  scale_x_discrete(limits = c("mouse", "rat", "monkey","human"))

acc_withfr = lm(tau ~ species + fr, data = acc)
acc_withfr  %>% summary()
acc_nofr = lm(tau ~ species, data = acc)
acc_nofr %>% summary()

anova(acc_withfr,acc_nofr)


# Amygdala

amyg %>% ggplot(aes(x=species,y=tau,color=brain_area,fill=brain_area)) +
  geom_flat_violin(position=position_nudge(x=0.2,y=0),adjust=2,alpha=0.5) + 
  geom_point(position=position_jitter(width=0.15),size=0.25) + coord_flip() +
  ggtitle('Amygdala - pop fit') +theme(plot.title = element_text(hjust = 0.5),legend.position='bottom') +
  scale_y_log10(breaks = scales::trans_breaks("log10", function(x) 10^x),
                labels = scales::trans_format("log10", scales::math_format(10^.x)),
                limits = c(9,1100)) +
  scale_x_discrete(limits = c("mouse", "rat", "monkey","human"))

amyg_withfr = lm(tau ~ species + fr, data = amyg)
amyg_withfr  %>% summary()
amyg_nofr = lm(tau ~ species, data = amyg)
amyg_nofr %>% summary()

anova(amyg_withfr,amyg_nofr)

# Hippocampus

hc %>% ggplot(aes(x=species,y=tau,color=brain_area,fill=brain_area)) +
  geom_flat_violin(position=position_nudge(x=0.2,y=0),adjust=2,alpha=0.5) + 
  geom_point(position=position_jitter(width=0.15),size=0.25) + coord_flip() +
  ggtitle('Hippocampus - pop fit') +theme(plot.title = element_text(hjust = 0.5),legend.position='bottom') +
  scale_y_log10(breaks = scales::trans_breaks("log10", function(x) 10^x),
                labels = scales::trans_format("log10", scales::math_format(10^.x)),
                limits = c(9,1100)) +
  scale_x_discrete(limits = c("mouse", "rat", "monkey","human"))

hc_withfr = lm(tau ~ species + fr, data = hc)
hc_withfr  %>% summary()
hc_nofr = lm(tau ~ species, data = hc)
hc_nofr %>% summary()

anova(hc_withfr,hc_nofr)

# Insula

insula %>% ggplot(aes(x=species,y=tau,color=brain_area,fill=brain_area)) +
  geom_flat_violin(position=position_nudge(x=0.2,y=0),adjust=2,alpha=0.5) + 
  geom_point(position=position_jitter(width=0.15),size=0.25) + coord_flip() +
  ggtitle('Insula - pop fit') +theme(plot.title = element_text(hjust = 0.5),legend.position='bottom') +
  scale_y_log10(breaks = scales::trans_breaks("log10", function(x) 10^x),
                labels = scales::trans_format("log10", scales::math_format(10^.x)),
                limits = c(9,1100)) +
  scale_x_discrete(limits = c("mouse", "rat", "monkey","human"))

# mPFC

mpfc %>% ggplot(aes(x=species,y=tau,color=brain_area,fill=brain_area)) +
  geom_flat_violin(position=position_nudge(x=0.2,y=0),adjust=2,alpha=0.5) + 
  geom_point(position=position_jitter(width=0.15),size=0.25) + coord_flip() +
  ggtitle('mPFC') +theme(plot.title = element_text(hjust = 0.5),legend.position='bottom') +
  scale_y_log10(breaks = scales::trans_breaks("log10", function(x) 10^x),
                labels = scales::trans_format("log10", scales::math_format(10^.x)),
                limits = c(9,1100)) +
  scale_x_discrete(limits = c("mouse", "rat", "monkey","human"))

mpfc_withfr = lm(tau ~ species + fr, data = mpfc)
mpfc_withfr  %>% summary()
mpfc_nofr = lm(tau ~ species, data = mpfc)
mpfc_nofr %>% summary()

anova(mpfc_withfr,mpfc_nofr)

# OFC

ofc %>% ggplot(aes(x=species,y=tau,color=brain_area,fill=brain_area)) +
  geom_flat_violin(position=position_nudge(x=0.2,y=0),adjust=2,alpha=0.5) + 
  geom_point(position=position_jitter(width=0.15),size=0.25) + coord_flip() +
  ggtitle('OFC - pop fit') +theme(plot.title = element_text(hjust = 0.5),legend.position='bottom') +
  scale_y_log10(breaks = scales::trans_breaks("log10", function(x) 10^x),
                labels = scales::trans_format("log10", scales::math_format(10^.x)),
                limits = c(9,1100)) +
  scale_x_discrete(limits = c("mouse", "rat", "monkey","human"))

ofc_withfr = lm(tau ~ species + fr, data = ofc)
ofc_withfr  %>% summary()
ofc_nofr = lm(tau ~ species, data = ofc)
ofc_nofr %>% summary()

anova(ofc_withfr,ofc_nofr)

# Striatum

str %>% ggplot(aes(x=species,y=tau,color=brain_area,fill=brain_area)) +
  geom_flat_violin(position=position_nudge(x=0.2,y=0),adjust=2,alpha=0.5) + 
  geom_point(position=position_jitter(width=0.15),size=0.25) + coord_flip() +
  ggtitle('Striatum - pop fit') +theme(plot.title = element_text(hjust = 0.5),legend.position='bottom') +
  scale_y_log10(breaks = scales::trans_breaks("log10", function(x) 10^x),
                labels = scales::trans_format("log10", scales::math_format(10^.x)),
                limits = c(9,1100)) +
  scale_x_discrete(limits = c("mouse", "rat", "monkey","human"))

## Within species just for fun

grouped_data %>% filter(species == 'mouse') %>% ggplot(aes(x=brain_area,y=tau,color=dataset,fill=dataset)) +
  geom_flat_violin(position=position_nudge(x=0.2,y=0),adjust=2,alpha=0.5) + 
  geom_point(position=position_jitter(width=0.15),size=0.25) + coord_flip() +
  ggtitle('Mouse - pop fits') +theme(plot.title = element_text(hjust = 0.5),legend.position='bottom') +
  scale_y_log10(breaks = scales::trans_breaks("log10", function(x) 10^x),
                labels = scales::trans_format("log10", scales::math_format(10^.x)),
                limits = c(9,1100))

grouped_data %>% filter(species == 'rat') %>% ggplot(aes(x=brain_area,y=tau,color=dataset,fill=dataset)) +
  geom_flat_violin(position=position_nudge(x=0.2,y=0),adjust=2,alpha=0.5) + 
  geom_point(position=position_jitter(width=0.15),size=0.25) + coord_flip() +
  ggtitle('Rat - pop fits') +theme(plot.title = element_text(hjust = 0.5),legend.position='bottom') +
  scale_y_log10(breaks = scales::trans_breaks("log10", function(x) 10^x),
                labels = scales::trans_format("log10", scales::math_format(10^.x)),
                limits = c(9,1100))

grouped_data %>% filter(species == 'monkey') %>% ggplot(aes(x=brain_area,y=tau,color=dataset,fill=dataset)) +
  geom_flat_violin(position=position_nudge(x=0.2,y=0),adjust=2,alpha=0.5) + 
  geom_point(position=position_jitter(width=0.15),size=0.25) + coord_flip() +
  ggtitle('Monkey - pop fits') +theme(plot.title = element_text(hjust = 0.5),legend.position='bottom') +
  scale_y_log10(breaks = scales::trans_breaks("log10", function(x) 10^x),
                labels = scales::trans_format("log10", scales::math_format(10^.x)),
                limits = c(9,1100))

grouped_data %>% filter(species == 'human') %>% ggplot(aes(x=brain_area,y=tau,color=dataset,fill=dataset)) +
  geom_flat_violin(position=position_nudge(x=0.2,y=0),adjust=2,alpha=0.5) + 
  geom_point(position=position_jitter(width=0.15),size=0.25) + coord_flip() +
  ggtitle('Human - pop fits') +theme(plot.title = element_text(hjust = 0.5),legend.position='bottom') +
  scale_y_log10(breaks = scales::trans_breaks("log10", function(x) 10^x),
                labels = scales::trans_format("log10", scales::math_format(10^.x)),
                limits = c(9,1100))

## Nice line graph

grouped_data %>% filter(brain_region %in% c('OFC','Hippocampus','Amygdala','ACC','mPFC')) %>% group_by(species,brain_region) %>%
  summarise(mean_tau = mean(tau), sd = sd(tau)) %>%
  ggplot(aes(x=reorder(brain_region,mean_tau,mean),y=mean_tau,color=species,fill=species)) +
  geom_point(size=3) + geom_line(aes(group=species))+
  geom_errorbar(aes(ymin=mean_tau-sd, ymax=mean_tau+sd), width=.1) + 
  xlab('Brain Region') +
  ylab('tau (ms)') + ggtitle('Population fit 1000x')
