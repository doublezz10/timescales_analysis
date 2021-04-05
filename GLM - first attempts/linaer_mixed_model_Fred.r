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

# Load in dataframe

df_fred = read.csv('~/Documents/timescales_analysis/GLM/fred_data_v3.csv')

df_fred = df_fred %>% mutate(species = factor(species),unitID = as.character(unitID))
df_fred = df_fred %>% rename(unit_id = unitID,
                             brain_area = area,
                             dataset = name,
                             fr = FR)

# Run everything all at once

lmer(tau ~ species * area + (1|fr) + (1|r2), data = df_fred) %>% summary()

# lmer(lat ~ species * area + (1|fr) + (1|r2), data = df_fred) %>% summary()

# Filter by brain area 

acc_fred = df_fred %>% filter(brain_area == 'acc' | brain_area == 'aca' | 
                      brain_area == 'dACC' | brain_area == 'scACC')

amyg_fred = df_fred %>% filter(brain_area == 'amygdala' | brain_area == 'bla' | 
                               brain_area == 'AMG' | brain_area == 'central')

hc_fred = df_fred %>% filter(brain_area == 'hippocampus' | brain_area == 'hippocampus2' |
                            brain_area == 'ca1' | brain_area == 'ca2' | 
                              brain_area == 'ca3' | brain_area == 'dg')

mpfc_fred =  df_fred %>% filter(brain_area == 'pl' | brain_area == 'ila' | 
                                brain_area == 'mpfc' | brain_area == 'mPFC')

ofc_fred = df_fred %>% filter(brain_area == 'ofc' | brain_area == 'orb' | 
                              brain_area == 'OFC')

premotor_fred = df_fred %>% filter(brain_area == 'PMd' | brain_area == 'preSMA')

str_fred = df_fred %>% filter(brain_area == 'ventralStriatum' | brain_area == 'Cd' | 
                              brain_area == 'PUT')

# Raincloud plots using Fred's data

df_fred %>% ggplot(aes(x=species,y=tau,color=dataset,fill=dataset)) +
  geom_flat_violin(position=position_nudge(x=0.2,y=0),adjust=2,alpha=0.7) + 
  geom_point(position=position_jitter(width=0.15),size=0.25) + coord_flip() +
  ggtitle('All Units - Fred') +theme(plot.title = element_text(hjust = 0.5),legend.position='bottom') +
  scale_y_log10(breaks = scales::trans_breaks("log10", function(x) 10^x),
                labels = scales::trans_format("log10", scales::math_format(10^.x)),
                limits = c(9,1100)) +
  scale_x_discrete(limits = c("mouse", "rat", "monkey","human"))

amyg_fred %>% ggplot(aes(x=species,y=tau,color=dataset,fill=dataset)) +
  geom_flat_violin(position=position_nudge(x=0.2,y=0),adjust=2,alpha=0.7) + 
  geom_point(position=position_jitter(width=0.15),size=0.25) + coord_flip() +
  ggtitle('Amygdala - Fred') +theme(plot.title = element_text(hjust = 0.5),legend.position='bottom') +
  scale_y_log10(breaks = scales::trans_breaks("log10", function(x) 10^x),
                labels = scales::trans_format("log10", scales::math_format(10^.x))) +
  scale_x_discrete(limits = c("mouse", "rat", "monkey","human"))

acc_fred %>% ggplot(aes(x=species,y=tau,color=dataset,fill=dataset)) +
  geom_flat_violin(position=position_nudge(x=0.2,y=0),adjust=2,alpha=0.7) + 
  geom_point(position=position_jitter(width=0.15),size=0.25) + coord_flip() +
  ggtitle('ACC - Fred') +theme(plot.title = element_text(hjust = 0.5),legend.position='bottom') +
  scale_y_log10(breaks = scales::trans_breaks("log10", function(x) 10^x),
                labels = scales::trans_format("log10", scales::math_format(10^.x))) +
  scale_x_discrete(limits = c("mouse", "rat", "monkey","human"))

hc_fred %>% ggplot(aes(x=species,y=tau,color=dataset,fill=dataset)) +
  geom_flat_violin(position=position_nudge(x=0.2,y=0),adjust=2,alpha=0.7) + 
  geom_point(position=position_jitter(width=0.15),size=0.25) + coord_flip() +
  ggtitle('Hippocampus - Fred') +theme(plot.title = element_text(hjust = 0.5),legend.position='bottom') +
  scale_y_log10(breaks = scales::trans_breaks("log10", function(x) 10^x),
                labels = scales::trans_format("log10", scales::math_format(10^.x)),
                limits = c(9,1100)) +
  scale_x_discrete(limits = c("mouse", "rat", "monkey","human"))

mpfc_fred %>% ggplot(aes(x=species,y=tau,color=dataset,fill=dataset)) +
  geom_flat_violin(position=position_nudge(x=0.2,y=0),adjust=2,alpha=0.7) + 
  geom_point(position=position_jitter(width=0.15),size=0.25) + coord_flip() +
  ggtitle('mPFC - Fred') +theme(plot.title = element_text(hjust = 0.5),legend.position='bottom') +
  scale_y_log10(breaks = scales::trans_breaks("log10", function(x) 10^x),
                labels = scales::trans_format("log10", scales::math_format(10^.x)),
                limits = c(9,1100)) +
  scale_x_discrete(limits = c("mouse", "rat", "monkey","human"))

ofc_fred %>% ggplot(aes(x=species,y=tau,color=dataset,fill=dataset)) +
  geom_flat_violin(position=position_nudge(x=0.2,y=0),adjust=2,alpha=0.7) + 
  geom_point(position=position_jitter(width=0.15),size=0.25) + coord_flip() +
  ggtitle('OFC- Fred') +theme(plot.title = element_text(hjust = 0.5),legend.position='bottom') +
  scale_y_log10(breaks = scales::trans_breaks("log10", function(x) 10^x),
                labels = scales::trans_format("log10", scales::math_format(10^.x)),
                limits = c(9,1100)) +
  scale_x_discrete(limits = c("mouse", "rat", "monkey","human"))

# Run models

lmer(tau ~ species + (1|fr), data = acc_fred) %>% summary()
lmer(tau ~ species + (1|fr), data = amyg_fred) %>% summary()
lmer(tau ~ species + (1|fr), data = hc_fred) %>% summary()
lmer(tau ~ species + (1|fr), data = mpfc_fred) %>% summary()
lmer(tau ~ species + (1|fr), data = ofc_fred) %>% summary()

lmer(tau ~ species + (1|fr), data = acc_fred) %>% emmeans(pairwise ~ species)
lmer(tau ~ species + (1|fr), data = amyg_fred) %>% emmeans(pairwise ~ species)
lmer(tau ~ species + (1|fr), data = hc_fred) %>% emmeans(pairwise ~ species)
lmer(tau ~ species + (1|fr), data = mpfc_fred) %>% emmeans(pairwise ~ species)
lmer(tau ~ species + (1|fr), data = ofc_fred) %>% emmeans(pairwise ~ species)

# Predict latency?

lmer(lat ~ species + (1|fr), data = acc_fred) %>% summary()
lmer(lat ~ species + (1|fr), data = amyg_fred) %>% summary()
lmer(lat ~ species + (1|fr), data = hc_fred) %>% summary()
lmer(lat ~ species + (1|fr), data = mpfc_fred) %>% summary()
lmer(lat ~ species + (1|fr), data = ofc_fred) %>% summary()

lmer(lat ~ species + (1|fr), data = acc_fred) %>% emmeans(pairwise ~ species)
lmer(lat ~ species + (1|fr), data = amyg_fred) %>% emmeans(pairwise ~ species)
lmer(lat ~ species + (1|fr), data = hc_fred) %>% emmeans(pairwise ~ species)
lmer(lat ~ species + (1|fr), data = mpfc_fred) %>% emmeans(pairwise ~ species)
lmer(lat ~ species + (1|fr), data = ofc_fred) %>% emmeans(pairwise ~ species)

