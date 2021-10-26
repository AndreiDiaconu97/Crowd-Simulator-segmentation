library(tidyverse)
library(rjson)
library(ggrepel)
library(ggpubr)
library("Cairo")
# CairoWin()

results_paths <- c(
    "pretrained_test_real",
    "pretrained_finetuning_COCO_100000",
    "pretrained_finetuning_50000", # is pretrained_finetuning Synth 100000
    "pretrained_finetuning_Mix_100000",
    "scratch_trained_COCO_100000",
    "scratch_trained_50000", # is scratch_trained Synth 100000
    "scratch_trained_Mix_100000"
)

tests_coco <- lapply(paste0(results_paths, "/evaluation_result_coco.json"), function(x) fromJSON(file = x) %>% as.data.frame()) %>% bind_rows()
tests_coco_filter_small <- lapply(paste0(results_paths, "/evaluation_result_coco_filter_small.json"), function(x) fromJSON(file = x) %>% as.data.frame()) %>% bind_rows()
tests_synthetic <- lapply(paste0(results_paths, "/evaluation_result_synthetic.json"), function(x) fromJSON(file = x) %>% as.data.frame()) %>% bind_rows()

colnames(tests_coco) <- paste("coco", colnames(tests_coco), sep = "_")
colnames(tests_coco_filter_small) <- paste("cocoFilterSmall", colnames(tests_coco_filter_small), sep = "_")
colnames(tests_synthetic) <- paste("synth", colnames(tests_synthetic), sep = "_")
# tests_coco$ID <- seq.int(nrow(tests_coco))
# tests_coco_filter_small$ID <- seq.int(nrow(tests_coco_filter_small))
# tests_synthetic$ID <- seq.int(nrow(tests_synthetic))

data <- bind_cols(tests_coco, tests_coco_filter_small, tests_synthetic)

res_labels <- c(
    "pretrained_test_real",
    "pretrained_finetuning_COCO",
    "pretrained_finetuning_Synth", # is pretrained_finetuning Synth 100000
    "pretrained_finetuning_Mix",
    "scratch_trained_COCO",
    "scratch_trained_Synth", # is scratch_trained Synth 100000
    "scratch_trained_Mix"
)

p_bbox_AP <- ggplot() +
    geom_point(data = data, aes(x = synth_bbox.AP, y = coco_bbox.AP, color = res_labels, shape = ">4 people")) +
    geom_point(data = data, aes(x = synth_bbox.AP, y = cocoFilterSmall_bbox.AP, color = res_labels, shape = ">4 people and no small ones")) +
    geom_line(aes(x = rep(data$synth_bbox.AP, each = 2), y = Map(c, data$coco_bbox.AP, data$cocoFilterSmall_bbox.AP) %>% unlist(), group = seq(1, nrow(data)) %>% rep(each = 2), color = res_labels %>% rep(each = 2)), show.legend = FALSE) +
    scale_y_continuous(breaks = seq(0, 100, by = 10), limits = c(0, 100)) +
    scale_x_continuous(breaks = seq(0, 100, by = 10), limits = c(0, 100)) +
    scale_shape_manual(name = "Selected COCO dataset for testing", values = c(1, 2)) +
    geom_text_repel(data = data, aes(x = synth_bbox.AP, y = coco_bbox.AP, label = paste0(res_labels, "\n[X=", round(data$synth_bbox.AP, 2), ", Y1=", round(data$coco_bbox.AP, 2), ", Y2=", round(data$cocoFilterSmall_bbox.AP, 2), "]")), box.padding = 2.5, segment.size = 0.2, segment.alpha = 0.5, segment.linetype = 3, size = 3) +
    theme_light() +
    coord_fixed() +
    guides(colour = FALSE) +
    theme(
        legend.position = c(0.22, 0.9),
        legend.background = element_rect(linetype = "solid", color = "black"),
        plot.title = element_text(hjust = 0.5, size = 20),
    ) +
    labs(
        title = "Bbox AP Scores",
        x = "Synthetic score",
        y = "Coco score"
    )

p_segm_AP <- ggplot() +
    geom_point(data = data, aes(x = synth_segm.AP, y = coco_segm.AP, color = res_labels, shape = ">4 people")) +
    geom_point(data = data, aes(x = synth_segm.AP, y = cocoFilterSmall_segm.AP, color = res_labels, shape = ">4 people and no small ones")) +
    geom_line(aes(x = rep(data$synth_segm.AP, each = 2), y = Map(c, data$coco_segm.AP, data$cocoFilterSmall_segm.AP) %>% unlist(), group = seq(1, nrow(data)) %>% rep(each = 2), color = res_labels %>% rep(each = 2)), show.legend = FALSE) +
    scale_y_continuous(breaks = seq(0, 100, by = 10), limits = c(0, 100)) +
    scale_x_continuous(breaks = seq(0, 100, by = 10), limits = c(0, 100)) +
    scale_shape_manual(name = "Selected COCO dataset for testing", values = c(1, 2)) +
    geom_text_repel(data = data, aes(x = synth_segm.AP, y = coco_segm.AP, label = paste0(res_labels, "\n[X=", round(data$synth_segm.AP, 2), ", Y1=", round(data$coco_segm.AP, 2), ", Y2=", round(data$cocoFilterSmall_segm.AP, 2), "]")), box.padding = 2.5, segment.size = 0.2, segment.alpha = 0.5, segment.linetype = 3, size = 3) +
    theme_light() +
    coord_fixed() +
    guides(colour = FALSE) +
    theme(
        legend.position = c(0.22, 0.9),
        legend.background = element_rect(linetype = "solid", color = "black"),
        plot.title = element_text(hjust = 0.5, size = 20)
    ) +
    labs(
        title = "Segm AP Scores",
        x = "Synthetic score",
        y = "Coco score"
    )

p_barplot_synth_bbox <- ggplot() +
    geom_bar(aes(x = res_labels %>% rep(6), y = as.numeric(unlist(data[29:34])), fill = sub(".*\\.", "", colnames(data[29:34])) %>% rep(each = nrow(data))), stat = "identity", position = "dodge") +
    scale_fill_manual(name = "Score", values = c("1", "2", "3", "4", "5", "6")) +
    scale_y_continuous(breaks = seq(0, 100, by = 10), limits = c(0, 100)) +
    theme_light() +
    theme(
        axis.text.x = element_text(angle = -8)
    )

p_barplot_synth_segm <- ggplot() +
    geom_bar(aes(x = res_labels %>% rep(6), y = as.numeric(unlist(data[36:41])), fill = colnames(data[36:41]) %>% rep(each = nrow(data))), stat = "identity", position = "dodge") +
    scale_fill_manual(name = "Score", values = c("1", "2", "3", "4", "5", "6")) +
    scale_y_continuous(breaks = seq(0, 100, by = 10), limits = c(0, 100)) +
    theme_light() +
    theme(
        axis.text.x = element_text(angle = -8),
        axis.title.y = element_text(size = 16, face = "bold")
    ) +
    labs(
        y = "Synth"
    )

p_barplot_coco_bbox <- ggplot() +
    geom_bar(aes(x = res_labels %>% rep(6), y = as.numeric(unlist(data[1:6])), fill = colnames(data[1:6]) %>% rep(each = nrow(data))), stat = "identity", position = "dodge") +
    scale_fill_manual(name = "Score", values = c("1", "2", "3", "4", "5", "6")) +
    scale_y_continuous(breaks = seq(0, 100, by = 10), limits = c(0, 100)) +
    theme_light() +
    theme(
        axis.text.x = element_text(angle = -8)
    )

p_barplot_coco_segm <- ggplot() +
    geom_bar(aes(x = res_labels %>% rep(6), y = as.numeric(unlist(data[8:13])), fill = colnames(data[8:13]) %>% rep(each = nrow(data))), stat = "identity", position = "dodge") +
    scale_fill_manual(name = "Score", values = c("1", "2", "3", "4", "5", "6")) +
    scale_y_continuous(breaks = seq(0, 100, by = 10), limits = c(0, 100)) +
    theme_light() +
    theme(
        axis.text.x = element_text(angle = -8),
        axis.title.y = element_text(size = 16, face = "bold")
    ) +
    labs(
        y = "Coco"
    )

p_barplot_cocoFilterSmall_bbox <- ggplot() +
    geom_bar(aes(x = res_labels %>% rep(6), y = as.numeric(unlist(data[15:20])), fill = colnames(data[15:20]) %>% rep(each = nrow(data))), stat = "identity", position = "dodge") +
    scale_fill_manual(name = "Score", values = c("1", "2", "3", "4", "5", "6")) +
    scale_y_continuous(breaks = seq(0, 100, by = 10), limits = c(0, 100)) +
    theme_light() +
    theme(
        axis.text.x = element_text(angle = -8),
        axis.title.x = element_text(size = 16, face = "bold")
    ) +
    labs(
        x = "Bbox"
    )

p_barplot_cocoFilterSmall_segm <- ggplot() +
    geom_bar(aes(x = res_labels %>% rep(6), y = as.numeric(unlist(data[22:27])), fill = colnames(data[22:27]) %>% rep(each = nrow(data))), stat = "identity", position = "dodge") +
    scale_fill_manual(name = "Score", values = c("1", "2", "3", "4", "5", "6")) +
    scale_y_continuous(breaks = seq(0, 100, by = 10), limits = c(0, 100)) +
    theme_light() +
    theme(
        axis.text.x = element_text(angle = -8),
        axis.title.x = element_text(size = 16, face = "bold"),
        axis.title.y = element_text(size = 16, face = "bold")
    ) +
    labs(
        y = "CocoFilterSmall",
        x = "Segm"
    )

p_barplot_TOT <- ggarrange(
    p_barplot_synth_bbox + rremove("ylab") + rremove("xlab"),
    p_barplot_synth_segm + rremove("xlab"),
    p_barplot_coco_bbox + rremove("ylab") + rremove("xlab"),
    p_barplot_coco_segm + rremove("xlab"),
    p_barplot_cocoFilterSmall_bbox + rremove("ylab"),
    p_barplot_cocoFilterSmall_segm,
    common.legend = TRUE, ncol = 2, nrow = 3 # legend = "none"
)
p_barplot_TOT_annotated <- annotate_figure(p_barplot_TOT,
    top = text_grob("Models Performance Overview", color = "black", face = "bold", size = 24),
    # bottom = text_grob("Data source: \n ToothGrowth data set",
    #     color = "blue",
    #     hjust = 1, x = 1, face = "italic", size = 10
    # ),
    # left = text_grob("Figure arranged using ggpubr", color = "green", rot = 90),
    # right = text_grob(bquote("Superscript: (" * kg ~ NH[3] ~ ha^-1 ~ yr^-1 * ")"), rot = 90),
    # fig.lab = "Figure 1", fig.lab.face = "bold"
)

# TODO: fix legend labels
# TODO: specify test dataset on Y axis
# TODO: better fit for x axis text

ggsave("plots/p_bbox_AP.png", p_bbox_AP, width = 7, height = 7)
ggsave("plots/p_segm_AP.png", p_segm_AP, width = 7, height = 7)
ggsave("plots/p_barplot_TOT.png", p_barplot_TOT_annotated)