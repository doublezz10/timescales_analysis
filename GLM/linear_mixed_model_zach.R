library(tidyverse)
library(lme4)
library(lmerTest)
library(ggplot2)
library(scales)
library(emmeans)

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

## Load in dataframe
df = read.csv('~/Documents/timescales_analysis/GLM/dataframe.csv')

## Run everything all at once

lmer(tau ~ species + brain_area + (1|fr), data = df) %>% summary()

## Filter by brain area - no r^2 or tau filtering

acc = df %>% filter(brain_area == 'acc' | brain_area == 'aca' | 
                      brain_area == 'dacc' | brain_area == 'scacc')

amyg = df %>% filter(brain_area == 'amyg' | brain_area == 'bla' | brain_area == 'centralNuc' |
                     brain_area == 'AMG')

hc = df %>% filter(brain_area == 'hc' | brain_area == 'ca1' | 
                     brain_area == 'ca2' | brain_area == 'ca3' | 
                     brain_area == 'dg')

ant_ins = df %>% filter(brain_area == 'lai' | brain_area == 'lamp')

mpfc =  df %>% filter(brain_area == 'pl' | brain_area == 'ila' | 
                      brain_area == 'mpfc')

ofc = df %>% filter(brain_area == 'ofc' | brain_area == 'orb' | brain_area == '11l' |
                    brain_area == '11m' | brain_area == '13m' | brain_area == '13l' |
                    brain_area == '13b' | brain_area == '12o')

str = df %>% filter(brain_area == 'vs' | brain_area == 'pu' | brain_area == 'cd')

# lmer(tau ~ species + (1|fr), data = acc) %>% summary()
# lmer(tau ~ species + (1|fr), data = amyg) %>% summary()
# lmer(tau ~ species + (1|fr), data = hc) %>% summary()
# lmer(tau ~ species + (1|fr), data = mpfc) %>% summary()
# lmer(tau ~ species + (1|fr), data = ofc) %>% summary()

## Do it all again but filter for r2 > 0.5 and 10 < tau < 1000

df_filtered = df %>% filter(r2 >= 0.5 & between(tau,10,1000) == TRUE)

acc_filtered = acc %>% filter(r2 >= 0.5 & between(tau,10,1000) == TRUE)

amyg_filtered = amyg %>% filter(r2 >= 0.5 & between(tau,10,1000) == TRUE)

hc_filtered = hc %>% filter(r2 >= 0.5 & between(tau,10,1000) == TRUE)

ant_ins_filtered = ant_ins %>% filter(r2 >= 0.5 & between(tau,10,1000) == TRUE)

mpfc_filtered = mpfc %>% filter(r2 >= 0.5 & between(tau,10,1000) == TRUE)

ofc_filtered = ofc %>% filter(r2 >= 0.5 & between(tau,10,1000) == TRUE)

str_filtered = str %>% filter(r2 >= 0.5 & between(tau,10,1000) == TRUE)

lmer(tau ~ species + (1|fr), data = df_filtered) %>% summary()

lmer(tau ~ species + (1|fr), data = acc_filtered) %>% summary()
lmer(tau ~ species + (1|fr), data = amyg_filtered) %>% summary()
lmer(tau ~ species + (1|fr), data = hc_filtered) %>% summary()
lmer(tau ~ species + (1|fr), data = mpfc_filtered) %>% summary()
lmer(tau ~ species + (1|fr), data = ofc_filtered) %>% summary()

## Try adding in R2 as random factor

lmer(tau ~ species + (1|fr) + (1|r2), data = acc_filtered) %>% summary()
lmer(tau ~ species + (1|fr) + (1|r2), data = amyg_filtered) %>% summary()
lmer(tau ~ species + (1|fr) + (1|r2), data = hc_filtered) %>% summary()
lmer(tau ~ species + (1|fr) + (1|r2), data = mpfc_filtered) %>% summary()
lmer(tau ~ species + (1|fr) + (1|r2), data = ofc_filtered) %>% summary()

## Pairwise comparisions

library(emmeans)

lmer(tau ~ species + (1|fr), data = acc_filtered) %>% emmeans(pairwise ~ species)
lmer(tau ~ species + (1|fr), data = amyg_filtered) %>% emmeans(pairwise ~ species)
lmer(tau ~ species + (1|fr), data = hc_filtered) %>% emmeans(pairwise ~ species)
lmer(tau ~ species + (1|fr), data = mpfc_filtered) %>% emmeans(pairwise ~ species)
lmer(tau ~ species + (1|fr), data = ofc_filtered) %>% emmeans(pairwise ~ species)

# Raincloud plots using filtered data - across species

df_filtered %>% ggplot(aes(x=species,y=tau,color=dataset,fill=dataset)) +
  geom_flat_violin(position=position_nudge(x=0.2,y=0),adjust=2,alpha=0.7) + 
  geom_point(position=position_jitter(width=0.15),size=0.25) + coord_flip() +
  ggtitle('All Units - Zach') +theme(plot.title = element_text(hjust = 0.5),legend.position='bottom') +
  scale_y_log10(breaks = scales::trans_breaks("log10", function(x) 10^x),
                labels = scales::trans_format("log10", scales::math_format(10^.x)),
                limits = c(9,1100)) +
  scale_x_discrete(limits = c("mouse", "rat", "monkey","human"))

amyg_filtered %>% ggplot(aes(x=species,y=tau,color=dataset,fill=dataset)) +
  geom_flat_violin(position=position_nudge(x=0.2,y=0),adjust=2,alpha=0.7) + 
  geom_point(position=position_jitter(width=0.15),size=0.25) + coord_flip() +
  ggtitle('Amygdala - Zach') +theme(plot.title = element_text(hjust = 0.5),legend.position='bottom') +
  scale_y_log10(breaks = scales::trans_breaks("log10", function(x) 10^x),
                labels = scales::trans_format("log10", scales::math_format(10^.x))) +
  scale_x_discrete(limits = c("mouse", "rat", "monkey","human"))

acc_filtered %>% ggplot(aes(x=species,y=tau,color=dataset,fill=dataset)) +
  geom_flat_violin(position=position_nudge(x=0.2,y=0),adjust=2,alpha=0.7) + 
  geom_point(position=position_jitter(width=0.15),size=0.25) + coord_flip() +
  ggtitle('ACC - Zach') +theme(plot.title = element_text(hjust = 0.5),legend.position='bottom') +
  scale_y_log10(breaks = scales::trans_breaks("log10", function(x) 10^x),
                labels = scales::trans_format("log10", scales::math_format(10^.x))) +
  scale_x_discrete(limits = c("mouse", "rat", "monkey","human"))

hc_filtered %>% ggplot(aes(x=species,y=tau,color=dataset,fill=dataset)) +
  geom_flat_violin(position=position_nudge(x=0.2,y=0),adjust=2,alpha=0.7) + 
  geom_point(position=position_jitter(width=0.15),size=0.25) + coord_flip() +
  ggtitle('Hippocampus - Zach') +theme(plot.title = element_text(hjust = 0.5),legend.position='bottom') +
  scale_y_log10(breaks = scales::trans_breaks("log10", function(x) 10^x),
                labels = scales::trans_format("log10", scales::math_format(10^.x)),
                limits = c(9,1100)) +
  scale_x_discrete(limits = c("mouse", "rat", "monkey","human"))

ant_ins_filtered %>% ggplot(aes(x=species,y=tau,color=dataset,fill=dataset)) +
  geom_flat_violin(position=position_nudge(x=0.2,y=0),adjust=2,alpha=0.7) + 
  geom_point(position=position_jitter(width=0.15),size=0.25) + coord_flip() +
  ggtitle('Ant Insula - Zach') +theme(plot.title = element_text(hjust = 0.5),legend.position='bottom') +
  scale_y_log10(breaks = scales::trans_breaks("log10", function(x) 10^x),
                labels = scales::trans_format("log10", scales::math_format(10^.x)),
                limits = c(9,1100)) +
  scale_x_discrete(limits = c("mouse", "rat", "monkey","human"))

mpfc_filtered %>% ggplot(aes(x=species,y=tau,color=dataset,fill=dataset)) +
  geom_flat_violin(position=position_nudge(x=0.2,y=0),adjust=2,alpha=0.7) + 
  geom_point(position=position_jitter(width=0.15),size=0.25) + coord_flip() +
  ggtitle('mPFC - Zach') +theme(plot.title = element_text(hjust = 0.5),legend.position='bottom') +
  scale_y_log10(breaks = scales::trans_breaks("log10", function(x) 10^x),
                labels = scales::trans_format("log10", scales::math_format(10^.x)),
                limits = c(9,1100)) +
  scale_x_discrete(limits = c("mouse", "rat", "monkey","human"))

ofc_filtered %>% ggplot(aes(x=species,y=tau,color=dataset,fill=dataset)) +
  geom_flat_violin(position=position_nudge(x=0.2,y=0),adjust=2,alpha=0.7) + 
  geom_point(position=position_jitter(width=0.15),size=0.25) + coord_flip() +
  ggtitle('OFC- Zach') +theme(plot.title = element_text(hjust = 0.5),legend.position='bottom') +
  scale_y_log10(breaks = scales::trans_breaks("log10", function(x) 10^x),
                labels = scales::trans_format("log10", scales::math_format(10^.x)),
                limits = c(9,1100)) +
  scale_x_discrete(limits = c("mouse", "rat", "monkey","human"))

str_filtered %>% ggplot(aes(x=species,y=tau,color=dataset,fill=dataset)) +
  geom_flat_violin(position=position_nudge(x=0.2,y=0),adjust=2,alpha=0.7) + 
  geom_point(position=position_jitter(width=0.15),size=0.25) + coord_flip() +
  ggtitle('Striatum- Zach') +theme(plot.title = element_text(hjust = 0.5),legend.position='bottom') +
  scale_y_log10(breaks = scales::trans_breaks("log10", function(x) 10^x),
                labels = scales::trans_format("log10", scales::math_format(10^.x)),
                limits = c(9,1100)) +
  scale_x_discrete(limits = c("mouse", "rat", "monkey","human"))

## Within species comparisons

mouse_df = df_filtered %>% filter(species=='mouse')
rat_df = df_filtered %>% filter(species=='rat')
monkey_df = df_filtered %>% filter(species=='monkey')
human_df = df_filtered %>% filter(species=='human')

lmer(tau ~ brain_area + (1|fr), data = mouse_df) %>% summary()
lmer(tau ~ brain_area + (1|fr), data = rat_df) %>% summary()
lmer(tau ~ brain_area + (1|fr), data = monkey_df) %>% summary()
lmer(tau ~ brain_area + (1|fr), data = human_df) %>% summary()

lmer(tau ~ brain_area + (1|fr), data = mouse_df) %>% emmeans(pairwise ~ brain_area)
lmer(tau ~ brain_area + (1|fr), data = rat_df) %>% emmeans(pairwise ~ brain_area)
lmer(tau ~ brain_area + (1|fr), data = monkey_df) %>% emmeans(pairwise ~ brain_area)
lmer(tau ~ brain_area + (1|fr), data = human_df) %>% emmeans(pairwise ~ brain_area)

mouse_df %>% ggplot(aes(x=brain_area,y=tau,color=dataset,fill=dataset)) +
  geom_flat_violin(position=position_nudge(x=0.2,y=0),adjust=2,alpha=0.7) + 
  geom_point(position=position_jitter(width=0.15),size=0.25) + coord_flip() +
  ggtitle('Mouse - Zach') +theme(plot.title = element_text(hjust = 0.5),legend.position='bottom') +
  scale_y_log10(breaks = scales::trans_breaks("log10", function(x) 10^x),
                labels = scales::trans_format("log10", scales::math_format(10^.x)),
                limits = c(9,1100))

rat_df %>% ggplot(aes(x=brain_area,y=tau,color=dataset,fill=dataset)) +
  geom_flat_violin(position=position_nudge(x=0.2,y=0),adjust=2,alpha=0.7) + 
  geom_point(position=position_jitter(width=0.15),size=0.25) + coord_flip() +
  ggtitle('Rat - Zach') +theme(plot.title = element_text(hjust = 0.5),legend.position='bottom') +
  scale_y_log10(breaks = scales::trans_breaks("log10", function(x) 10^x),
                labels = scales::trans_format("log10", scales::math_format(10^.x)),
                limits = c(9,1100))

monkey_df %>% ggplot(aes(x=brain_area,y=tau,color=dataset,fill=dataset)) +
  geom_flat_violin(position=position_nudge(x=0.2,y=0),adjust=2,alpha=0.7) + 
  geom_point(position=position_jitter(width=0.15),size=0.25) + coord_flip() +
  ggtitle('Monkey - Zach') +theme(plot.title = element_text(hjust = 0.5),legend.position='bottom') +
  scale_y_log10(breaks = scales::trans_breaks("log10", function(x) 10^x),
                labels = scales::trans_format("log10", scales::math_format(10^.x)),
                limits = c(9,1100))

human_df %>% ggplot(aes(x=brain_area,y=tau,color=dataset,fill=dataset)) +
  geom_flat_violin(position=position_nudge(x=0.2,y=0),adjust=2,alpha=0.7) + 
  geom_point(position=position_jitter(width=0.15),size=0.25) + coord_flip() +
  ggtitle('Human - Zach') +theme(plot.title = element_text(hjust = 0.5),legend.position='bottom') +
  scale_y_log10(breaks = scales::trans_breaks("log10", function(x) 10^x),
                labels = scales::trans_format("log10", scales::math_format(10^.x)),
                limits = c(9,1100))

## Effects of filtering

# amyg_f <- amyg_filtered %>% mutate(filtered = 'yes')
# amyg_nof <- amyg %>% mutate(filtered = 'no')
# 
# all_amyg = rbind(amyg_nof,amyg_f)
# 
# all_amyg %>% ggplot(aes(x=species,y=tau,color=filtered)) +
#   geom_split_violin() + scale_y_log10(breaks = scales::trans_breaks("log10", function(x) 10^x),
#                                       labels = scales::trans_format("log10", scales::math_format(10^.x))) +
#   ggtitle('Amygdala',subtitle='Zach') + theme(plot.title = element_text(hjust = 0.5),plot.subtitle = element_text(hjust = 0.5))
# 
# acc_f <- acc_filtered %>% mutate(filtered = 'yes')
# acc_nof <- acc %>% mutate(filtered = 'no')
# 
# all_acc = rbind(acc,acc_f)
# 
# all_acc %>% ggplot(aes(x=species,y=tau,color=filtered)) +
#   geom_split_violin() + scale_y_log10(breaks = scales::trans_breaks("log10", function(x) 10^x),
#                                       labels = scales::trans_format("log10", scales::math_format(10^.x))) +
#   ggtitle('ACC',subtitle='Zach') + theme(plot.title = element_text(hjust = 0.5),plot.subtitle = element_text(hjust = 0.5))
# 
# hc_f <- hc_filtered %>% mutate(filtered = 'yes')
# hc_nof <- hc %>% mutate(filtered = 'no')
# 
# all_hc = rbind(hc_f,hc_nof)
# 
# all_hc %>% ggplot(aes(x=species,y=tau,color=filtered)) +
#   geom_split_violin() + scale_y_log10(breaks = scales::trans_breaks("log10", function(x) 10^x),
#                                       labels = scales::trans_format("log10", scales::math_format(10^.x))) +
#   ggtitle('Hippocampus',subtitle='Zach') + theme(plot.title = element_text(hjust = 0.5),plot.subtitle = element_text(hjust = 0.5))
# 
# mpfc_f <- mpfc_filtered %>% mutate(filtered = 'yes')
# mpfc_nof <- mpfc %>% mutate(filtered = 'no')
# 
# all_mpfc = rbind(mpfc_f,mpfc_nof)
# 
# all_mpfc %>% ggplot(aes(x=species,y=tau,color=filtered)) +
#   geom_split_violin() + scale_y_log10(breaks = scales::trans_breaks("log10", function(x) 10^x),
#                                       labels = scales::trans_format("log10", scales::math_format(10^.x))) +
#   ggtitle('mPFC',subtitle='Zach') + theme(plot.title = element_text(hjust = 0.5),plot.subtitle = element_text(hjust = 0.5))
# 
# ofc_f <- ofc_filtered %>% mutate(filtered = 'yes')
# ofc_nof <- ofc %>% mutate(filtered = 'no')
# 
# all_ofc = rbind(ofc_nof,ofc_f)
# 
# all_ofc %>% ggplot(aes(x=species,y=tau,color=filtered)) +
#   geom_split_violin() + scale_y_log10(breaks = scales::trans_breaks("log10", function(x) 10^x),
#                                       labels = scales::trans_format("log10", scales::math_format(10^.x))) +
#   ggtitle('OFC',subtitle='Zach') + theme(plot.title = element_text(hjust = 0.5),plot.subtitle = element_text(hjust = 0.5))
