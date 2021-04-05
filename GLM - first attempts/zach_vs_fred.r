library(tidyverse)
library(lme4)
library(lmerTest)
library(ggplot2)
library(scales)
library(emmeans)

# Load in dataframes
df = read.csv('~/Documents/timescales_analysis/GLM/dataframe.csv')

# Filter
df_filtered = df %>% filter(r2 >= 0.5 & between(tau,10,1000) == TRUE)

df_filtered <- df_filtered %>% mutate(method='Zach')

acc_filtered = df_filtered %>% filter(brain_area == 'acc' | brain_area == 'aca' | 
                                        brain_area == 'dacc' | brain_area == 'scacc')

amyg_filtered = df_filtered %>% filter(brain_area == 'amyg' | brain_area == 'bla' | brain_area == 'centralNuc')

hc_filtered = df_filtered %>% filter(brain_area == 'hc' | brain_area == 'ca1' | 
                                       brain_area == 'ca2' | brain_area == 'ca3' | 
                                       brain_area == 'dg')

mpfc_filtered =  df_filtered %>% filter(brain_area == 'pl' | brain_area == 'ila' | 
                                          brain_area == 'mpfc')

ofc_filtered = df_filtered %>% filter(brain_area == 'ofc' | brain_area == 'orb')

# Add Fred's dataset
df_fred = read.csv('~/Documents/timescales_analysis/GLM/fred_data_v2.csv')

df_fred = df_fred %>% mutate(species = factor(species),unitID = as.character(unitID))
df_fred = df_fred %>% rename(unit_id = unitID,
                             brain_area = area,
                             dataset = name,
                             fr = FR,
                             a = A,
                             b = B)

df_fred <- df_fred %>% mutate(method='Fred')

df_fred = df_fred %>% select(dataset,species,brain_area,unit_id,tau,fr,r2,a,b,method)

acc_fred = df_fred %>% filter(brain_area == 'acc' | brain_area == 'aca' | 
                                brain_area == 'dACC' | brain_area == 'scACC')

amyg_fred = df_fred %>% filter(brain_area == 'amygdala' | brain_area == 'bla')

hc_fred = df_fred %>% filter(brain_area == 'hippocampus' | brain_area == 'ca1' | 
                               brain_area == 'ca2' | brain_area == 'ca3' | 
                               brain_area == 'dg')

mpfc_fred =  df_fred %>% filter(brain_area == 'pl' | brain_area == 'ila' | 
                                  brain_area == 'mpfc')

ofc_fred = df_fred %>% filter(brain_area == 'ofc' | brain_area == 'orb')

# Make split violin plots comparing distributions between methods

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

# Split violin plots

both_amyg <- rbind(amyg_filtered,amyg_fred)
# 
both_amyg %>% ggplot(aes(x=species,y=tau,color=method)) +
   geom_split_violin() + scale_y_log10(breaks = scales::trans_breaks("log10", function(x) 10^x),
                                       labels = scales::trans_format("log10", scales::math_format(10^.x))) +
   ggtitle('Amygdala') + theme(plot.title = element_text(hjust = 0.5))
 
both_acc <- rbind(acc_filtered,acc_fred)
# 
# both_acc %>% ggplot(aes(x=species,y=tau,color=method)) +
#   geom_split_violin() + scale_y_log10(breaks = scales::trans_breaks("log10", function(x) 10^x),
#                                       labels = scales::trans_format("log10", scales::math_format(10^.x))) +
#   ggtitle('ACC') + theme(plot.title = element_text(hjust = 0.5))
# 
both_hc <- rbind(hc_filtered,hc_fred)
# 
# both_hc %>% ggplot(aes(x=species,y=tau,color=method)) +
#   geom_split_violin() + scale_y_log10(breaks = scales::trans_breaks("log10", function(x) 10^x),
#                                       labels = scales::trans_format("log10", scales::math_format(10^.x))) +
#   ggtitle('Hippocampus') + theme(plot.title = element_text(hjust = 0.5))
# 
both_mpfc <- rbind(mpfc_filtered,mpfc_fred)
# 
# both_mpfc %>% ggplot(aes(x=species,y=tau,color=method)) +
#   geom_split_violin() + scale_y_log10(breaks = scales::trans_breaks("log10", function(x) 10^x),
#                                       labels = scales::trans_format("log10", scales::math_format(10^.x))) +
#   ggtitle('mPFC') + theme(plot.title = element_text(hjust = 0.5))
# 
both_ofc <- rbind(ofc_filtered,ofc_fred)
# 
# both_ofc %>% ggplot(aes(x=species,y=tau,color=method)) +
#   geom_split_violin() + scale_y_log10(breaks = scales::trans_breaks("log10", function(x) 10^x),
#                                       labels = scales::trans_format("log10", scales::math_format(10^.x))) +
#   ggtitle('OFC') + theme(plot.title = element_text(hjust = 0.5))

# Raincloud plots

sample_size = both_amyg %>% group_by(species,method) %>% summarize(num=n())

both_amyg %>% left_join(sample_size) %>% mutate(myaxis = paste0(method, "\n", "n=", num)) %>% 
  ggplot(aes(x=species,y=tau,color=method,fill=method)) +
  geom_flat_violin(position=position_nudge(x=0.2,y=0),adjust=2,alpha=0.7) + 
  geom_point(position=position_jitter(width=0.15),size=0.25) + coord_flip() +
  ggtitle('Amygdala') +theme(plot.title = element_text(hjust = 0.5),legend.position='bottom') +
  scale_y_log10(breaks = scales::trans_breaks("log10", function(x) 10^x),
                labels = scales::trans_format("log10", scales::math_format(10^.x)),
                limits = c(9,1100)) +
  scale_x_discrete(limits = c("mouse", "rat", "monkey","human"))

both_acc %>% ggplot(aes(x=species,y=tau,color=method,fill=method)) +
  geom_flat_violin(position=position_nudge(x=0.2,y=0),adjust=2,alpha=0.7) + 
  geom_point(position=position_jitter(width=0.15),size=0.25) + coord_flip() +
  ggtitle('ACC') +theme(plot.title = element_text(hjust = 0.5),legend.position='bottom') +
  scale_y_log10(breaks = scales::trans_breaks("log10", function(x) 10^x),
                labels = scales::trans_format("log10", scales::math_format(10^.x)),
                limits = c(9,1100)) +
  scale_x_discrete(limits = c("mouse", "rat", "monkey","human"))

both_hc %>% ggplot(aes(x=species,y=tau,color=method,fill=method)) +
  geom_flat_violin(position=position_nudge(x=0.2,y=0),adjust=2,alpha=0.7) + 
  geom_point(position=position_jitter(width=0.15),size=0.25) + coord_flip() +
  ggtitle('Hippocampus') +theme(plot.title = element_text(hjust = 0.5),legend.position='bottom') +
  scale_y_log10(breaks = scales::trans_breaks("log10", function(x) 10^x),
                labels = scales::trans_format("log10", scales::math_format(10^.x)),
                limits = c(9,1100)) +
  scale_x_discrete(limits = c("mouse", "rat", "monkey","human"))

both_mpfc %>% ggplot(aes(x=species,y=tau,color=method,fill=method)) +
  geom_flat_violin(position=position_nudge(x=0.2,y=0),adjust=2,alpha=0.7) + 
  geom_point(position=position_jitter(width=0.15),size=0.25) + coord_flip() +
  ggtitle('mPFC') +theme(plot.title = element_text(hjust = 0.5),legend.position='bottom') +
  scale_y_log10(breaks = scales::trans_breaks("log10", function(x) 10^x),
                labels = scales::trans_format("log10", scales::math_format(10^.x)),
                limits = c(9,1100)) +
  scale_x_discrete(limits = c("mouse", "rat", "monkey","human"))

both_ofc %>% ggplot(aes(x=species,y=tau,color=method,fill=method)) +
  geom_flat_violin(position=position_nudge(x=0.2,y=0),adjust=2,alpha=0.7) + 
  geom_point(position=position_jitter(width=0.15),size=0.25) + coord_flip() +
  ggtitle('OFC') +theme(plot.title = element_text(hjust = 0.5),legend.position='bottom') +
  scale_y_log10(breaks = scales::trans_breaks("log10", function(x) 10^x),
                labels = scales::trans_format("log10", scales::math_format(10^.x)),
                limits = c(9,1100)) +
  scale_x_discrete(limits = c("mouse", "rat", "monkey","human"))

# Linear models

lmer(tau ~ species * method + (1|fr), data = both_acc) %>% summary()
lmer(tau ~ species * method + (1|fr), data = both_amyg) %>% summary()
lmer(tau ~ species * method + (1|fr), data = both_hc) %>% summary()
lmer(tau ~ species * method + (1|fr), data = both_mpfc) %>% summary()
lmer(tau ~ species * method + (1|fr), data = both_ofc) %>% summary()

lmer(tau ~ species * method + (1|fr) + (1|r2), data = both_acc) %>% summary()
lmer(tau ~ species * method + (1|fr) + (1|r2), data = both_amyg) %>% summary()
lmer(tau ~ species * method + (1|fr) + (1|r2), data = both_hc) %>% summary()
lmer(tau ~ species * method + (1|fr) + (1|r2), data = both_mpfc) %>% summary()
lmer(tau ~ species * method + (1|fr) + (1|r2), data = both_ofc) %>% summary()

# Pairwise comparisions

lmer(tau ~ species * method + (1|fr), data = both_acc) %>% emmeans(pairwise ~ species)
lmer(tau ~ species * method + (1|fr), data = both_amyg) %>% emmeans(pairwise ~ species)
lmer(tau ~ species * method + (1|fr), data = both_hc) %>% emmeans(pairwise ~ species)
lmer(tau ~ species * method + (1|fr), data = both_mpfc) %>% emmeans(pairwise ~ species)
lmer(tau ~ species * method + (1|fr), data = both_ofc) %>% emmeans(pairwise ~ species)

# add all interaction terms to comparison

lmer(tau ~ species * method + (1|fr), data = both_acc) %>% emmeans(pairwise ~ species * method)
lmer(tau ~ species * method + (1|fr), data = both_amyg) %>% emmeans(pairwise ~ species * method)
lmer(tau ~ species * method + (1|fr), data = both_hc) %>% emmeans(pairwise ~ species * method)
lmer(tau ~ species * method + (1|fr), data = both_mpfc) %>% emmeans(pairwise ~ species * method)
lmer(tau ~ species * method + (1|fr), data = both_ofc) %>% emmeans(pairwise ~ species * method)

# Plot pairwise comparisons

amyg_emm = lmer(tau ~ species * method + (1|fr), data = both_amyg) %>% emmeans(~species * method)
pwpp(amyg_emm, comparisons = TRUE) + ggtitle('Amygdala')
plot(amyg_emm, comparisons = TRUE) + ggtitle('Amygdala')

acc_emm = lmer(tau ~ species * method + (1|fr), data = both_acc) %>% emmeans(~species * method)
pwpp(acc_emm, comparisons = TRUE) + ggtitle('ACC')

hc_emm = lmer(tau ~ species * method + (1|fr), data = both_hc) %>% emmeans(~species * method)
pwpp(hc_emm, comparisons = TRUE) + ggtitle('Hippocampus')

mpfc_emm = lmer(tau ~ species * method + (1|fr), data = both_mpfc) %>% emmeans(~species * method)
pwpp(mpfc_emm, comparisons = TRUE) + ggtitle('mPFC')

ofc_emm = lmer(tau ~ species * method + (1|fr), data = both_ofc) %>% emmeans(~species * method)
pwpp(ofc_emm, comparisons = TRUE) + ggtitle('OFC')
