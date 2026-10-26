library(xgboost)
library(readr)
library(glue)
library(ggplot2)
library(reshape2)
library(dplyr)
library(jsonlite)
library(paletteer) # scale_color_paletteer_c
library(grid) #rasterGrob
library(magick) # image_read, image_scale
library(patchwork)

library(RColorBrewer)
library(jpeg)
library(cowplot)

library(docopt)
library(jpeg)


library(parallel)

"infer_risk.R: takes morphometric features as input for risk inference with an XGBoost model
Usage:
  infer_risk.R --sample_name <NAME> --glioma_subtype <NAME> --classifier_model <NAME> --vit_model <NAME> --risk_model <NAME> --risk_cutoff <NUM> --threads <NUM>
  infer_risk.R --sample_name=<NAME> --glioma_subtype=<NAME> --classifier_model=<NAME> --vit_model <NAME> --risk_model=<NAME> --risk_cutoff=<NUM> --threads=<NUM>

Options:
  --sample_name NAME i.e. TCGA-06-0143-01Z-00-DX1
  --glioma_subtype NAME IDH-codel, IDH-non-codel, IDHwt
  --classifier_model NAME i.e. glial_histology_v3
  --vit_model NAME i.e. prov_gigapath
  --risk_model NAME i.e. xgb_risk_lgggbm_IDHwt_v4
  --risk_cutoff i.e. 0.465
  --threads NUM number of threads
  --help                show this help text
" -> doc
opt <- docopt(doc)



#opt<-list(study="GBM_havoc_paper_k6_px512_mpp252_kumar", classifier_model="glial_histology_v3", vit_model="prov_gigapath", risk_model="risk_glioma_v3", threads=6)
#opt<-list(sample_name="TCGA-06-0143-01Z-00-DX1", glioma_subtype="IDHwt", classifier_model="glial_histology_v3", vit_model="prov_gigapath", risk_model='xgb_risk_lgggbm_IDHwt_v4', risk_cutoff=0.6909994, threads=6)

study<-"figure4_k6_px512_mpp252_kumar"
sample.name<-opt$sample_name
glioma.subtype<-opt$glioma_subtype
classifier.model<-opt$classifier_model
risk.model<-opt$risk_model
threads=as.numeric(opt$threads)
vit.model<-opt$vit_model
risk.cutoff<-as.numeric(opt$risk_cutoff)

#dir.model<-file.path('/data/docker/havoc-profiler-main/model_XGBoost')
dir.model<-file.path("/app/model_XGBoost")
dir.base<-file.path("/data", "havoc_profiler")

if(!dir.exists(file.path(dir.base, "risk", study, sample.name))){dir.create(file.path(dir.base, "risk", study, sample.name))}
if(!dir.exists(file.path(dir.base, "risk", study, sample.name, glue("plots_{risk.model}")))){dir.create(file.path(dir.base, "risk", study, sample.name, glue("plots_{risk.model}")))}

# original model file: /data/computer_vision12_features_umap/results/12c_xgb_folds_mini.rds
xgb.folds<-readRDS(file.path(dir.model, glue("{risk.model}.rds")))
list.xgb<-lapply(xgb.folds, function(f){f$xgb.model})

df.classes<-fromJSON(file.path(dir.model, glue("xgb_{classifier.model}.json")))
#df.classes<-fromJSON(file.path("/data/docker/havoc-profiler-main/model_XGBoost", glue("xgb_{classifier.model}.json")))


dat.region<-suppressMessages(read_csv(file.path(dir.base, "classifier_region", paste0("class_region_", study, "_", classifier.model, ".csv"))))
dat.region$top_class_name[dat.region$top_class_name=="Glial_histology" & dat.region$top_class_score<0.33]<-"Other"


features<-data.frame(suppressMessages(read_csv(file=file.path(dir.base, "dlfv_region", glue("dlfv_region_{study}_{vit.model}.csv")))))
# the batch workflow merges the dlfv_region file by study; we keep the current sample only
is.sample.curr<-sapply(features$havoc_region, function(curr){
  grepl(sample.name, curr)
})
features<-features[is.sample.curr,]

rownames(features)<-features[,1]

is.tumor<-sapply(rownames(features), function(curr){
  r<-which(dat.region$region==curr)
  return(dat.region$top_class_name[r] %in% df.classes$class_name[df.classes$is_tumor])
})
features<-features[is.tumor,]

features.sel<-as.matrix(data.frame(features[,colnames(features) %in% list.xgb[[1]]$feature_names]))
identical(colnames(features.sel), list.xgb[[1]]$feature_names)

if(nrow(features.sel)>0){
  pred<-sapply(1:length(list.xgb), function(pos){
    predict(list.xgb[[pos]], features.sel)
  })
  if(nrow(features.sel)>1){ # sample contains more than one region
    pred<-rowMeans(pred)
  }else{
    pred<-mean(pred)
  }
  names(pred)<-rownames(features.sel)
  
  no.tiles.region<-sapply(rownames(features.sel), function(curr){
    r<-which(dat.region$region==curr)
    return(dat.region$region_tiles[r])
  })
  pred.weighed<-sum(pred * no.tiles.region) / sum(no.tiles.region)
  # pred.merge is used to save the values externally; to used for plot generation
  pred.merge<-data.frame(sample=rep(sample.name, length(pred)+1), region=c(names(pred), paste0(sample.name,"_weighed")), risk=as.numeric(c(pred, pred.weighed)))
}else{
  pred<-NA
  pred.weighed<-NA
  pred.merge<-data.frame(sample=sample.name, region=paste0(sample.name,"_weighed"), risk=NA)
}

write.csv(pred.merge, file.path(dir.base, "risk", study, sample.name, glue("{sample.name}_risk.csv")))

######################################################

dat.havoc<-suppressMessages(read_csv(file.path(dir.base, "havoc", study, sample.name, paste0(sample.name, "_cluster_info_df.csv"))))
kval<-length(unique(dat.havoc[,1, drop=TRUE]))

region.name<-as.character(sapply(dat.havoc$Coor, function(tile){
  region.curr<-dat.havoc[,paste0("Cluster_color_name_", kval), drop=TRUE][dat.havoc$Coor==tile]
  return(paste0(sample.name, "_", region.curr))
}))

dat.havoc<-cbind(dat.havoc, region_name=region.name)

reviewed.class<-sapply(dat.havoc$Coor, function(tile){
  region.curr<-dat.havoc[,paste0("Cluster_color_name_", kval), drop=TRUE][dat.havoc$Coor==tile]
  new.anno<-dat.region$top_class_name[dat.region$region==paste0(sample.name,"_", region.curr)]
  return(new.anno)
})

dat.dim<-read.table(file.path(dir.base, "havoc", study, sample.name, paste0(sample.name, "_dimensions.txt")))
dimensions<-list(height=dat.dim$V2[dat.dim$V1=="height"],
                 width=dat.dim$V2[dat.dim$V1=="width"],
                 magnification=dat.dim$V2[dat.dim$V1=="magnification"])

px.w<-dat.dim$V2[dat.dim$V1=="width"]
px.h<-dat.dim$V2[dat.dim$V1=="height"]
magnification=dat.dim$V2[dat.dim$V1=="magnification"]


scaling.factor<-magnification/2.5 # i.e. 40/2.5=16, or 20/2.5=8

coor.raw<-colsplit(gsub(" ","",gsub("]","",gsub("\\[","",dat.havoc$Coor))),",",c("x1","y1","x2","y2"))
coor.2_5x<-data.frame(apply(coor.raw,2, function(x) as.numeric(round(x/scaling.factor,0))))
tile.merge<-data.frame(tile=dat.havoc$Coor, coor.2_5x, tile.group=reviewed.class, region_name=dat.havoc$region_name)
tile.merge<-tile.merge[tile.merge$tile.group!="Blank space",]

color.scheme<-df.classes$color_scheme
names(color.scheme)<-df.classes$class_name
color.scheme<-color.scheme[color.scheme!="exclude"]


# not sure why plot limits require px.w/(2*scaling.factor)
p.class<-ggplot(data=tile.merge, aes(xmin=x1, xmax=x2, ymin=y1, ymax=y2, color=tile.group)) +
  geom_rect(fill="white") +
  #scale_x_continuous(expand=c(0,0), limits=c(0, px.w/(2*scaling.factor))) +
  #scale_y_reverse(expand=c(0,0), limits=c(px.h/(2*scaling.factor), 0)) +
  scale_y_reverse() +
  theme(legend.position = "none") +
  scale_color_manual(values = color.scheme) +
  theme_void() +
  theme(panel.border = element_rect(colour = "grey90", fill=NA, linewidth=1)) +
  labs(title=paste0("Classifier: ", classifier.model)) +
  theme(legend.position = "bottom", legend.title=element_blank()) +
  theme(plot.margin = unit(c(0.1,0.1,0.1,0.1), "cm"), 
        plot.title = element_text(hjust = 0.5)) +
  coord_fixed() # preserves aspect ratio (prevents "stretching")



fname.heatmap<-list.files(file.path(dir.base, "havoc", study, sample.name), pattern="_colortiled")
#img <- readJPEG(file.path(dir.base, "havoc", study, sample.name, fname.heatmap))  # use readJPEG() for .jpg files
img <- image_read(file.path(dir.base, "havoc", study, sample.name, fname.heatmap))
img_small <- image_scale(img, "10%") # downsizing prevents aberrations during raster
g <- rasterGrob(img_small, interpolate = TRUE)
p.heatmap<-ggplot() +
  annotation_custom(g) +
  theme_void() +
  labs(title="Histomorphic regions") +
  theme(plot.margin = unit(c(0.1,0.1,0.1,0.1), "cm"),
        plot.title = element_text(hjust = 0.5))

fname.thumb<-list.files(file.path(dir.base, "havoc", study, sample.name), pattern="_thumbnail")
img.thumb <- readJPEG(file.path(dir.base, "havoc", study, sample.name, fname.thumb))  # use readJPEG() for .jpg files
g <- rasterGrob(img.thumb, interpolate = TRUE)
p.thumb<-ggplot() +
  annotation_custom(g) +
  theme_void() +
  labs(title="H&E") +
  theme(plot.margin = unit(c(0.1,0.1,0.1,0.1), "cm"), 
        plot.title = element_text(hjust = 0.5))

##########################################################
############ risk heatmap + color scale bar ##############
##########################################################
# assigns a risk score to each tile, by region; a bit too convoluted, but it works
tile.merge.tumor<-tile.merge[tile.merge$tile.group %in% df.classes$class_name[df.classes$is_tumor],]
if(nrow(tile.merge.tumor)>0){
  list.res<-lapply(1:length(pred), function(i){
    tmp<-rep(NA, nrow(tile.merge.tumor))
    tmp[tile.merge.tumor$region_name==names(pred)[i]]<-pred[i]
    return(tmp)
  })
  res<-suppressMessages(t(bind_cols(list.res)))
  region.score<-as.numeric(sapply(1:ncol(res), function(c){
    v<-res[,c]
    return(v[!is.na(v)])
  }))
  tile.merge.tumor<-cbind(tile.merge.tumor, region_score=region.score)
  tile.merge.tumor$region_score[tile.merge.tumor$region_score>2.5]<-2.5 # cap very high values to color scale upper limit
  
  tile.merge.other<-cbind(tile.merge[!tile.merge$tile.group %in% df.classes$class_name[df.classes$is_tumor],])
  # This allows tables with zero rows (all the regions are tumor)
  tile.merge.other<-cbind(tile.merge.other, Tissue_type=rep("Non-tumor", nrow(tile.merge.other)))
  
  scale.clamp   <- 1.5           # Color scale upper anchor
  threshold_val <- risk.cutoff   # Script parameter (color scale midpoint)
  
  # Calculate free maximum for the y-axis (minimum 1.5)
  all_risk_vals <- pred.merge$risk[!is.na(pred.merge$risk)]
  y_axis_max    <- max(c(scale.clamp, all_risk_vals), na.rm = TRUE)
  
  dot_vals <- pred.merge$risk[!grepl("_weighed", pred.merge$region)]
  
  # Plot points at their true positions without clipping
  df_dots <- data.frame(
    y = dot_vals,
    x = 1,
    type = "regional_risk"
  )
  df_dots <- rbind(df_dots,
                   data.frame(y=pred.merge$risk[grepl("_weighed", pred.merge$region)], x=1, type="Weighed_mean"))
  
  base_breaks <- seq(0, scale.clamp, 0.5)
  if (threshold_val > 0.425 && threshold_val < 0.575) {
    base_breaks <- base_breaks[abs(base_breaks - 0.5) > 1e-4]
  }
  
  # Add ticks above 1.5 if the axis extends beyond 1.5
  if (y_axis_max > scale.clamp) {
    step_above <- if (y_axis_max <= 3) 0.5 else if (y_axis_max <= 6) 1 else 2
    first_break <- ceiling((scale.clamp + 1e-4) / step_above) * step_above
    if (first_break <= y_axis_max) {
      breaks_above <- seq(first_break, y_axis_max, by = step_above)
    } else {
      breaks_above <- numeric(0)
    }
  } else {
    breaks_above <- numeric(0)
  }
  
  all_breaks <- sort(unique(c(threshold_val, base_breaks, breaks_above)))
  
  # Dynamic labels reflecting threshold without artificial clamping text
  all_labels <- sapply(all_breaks, function(v) {
    if (abs(v - threshold_val) < 1e-4) {
      paste0(format(round(v, 2), nsmall = 2), " (Threshold)")
    } else {
      format(round(v, 2), nsmall = 2)
    }
  })
  
  p.risk.heatmap <- ggplot() +
    geom_rect(
      data = tile.merge.tumor,
      aes(xmin = x1, xmax = x2, ymin = y1, ymax = y2, color = region_score),
      fill = "white"
    ) +
    geom_rect(
      data = tile.merge.other,
      aes(xmin = x1, xmax = x2, ymin = y1, ymax = y2, fill = "Not Diffuse Glioma"),
      color = "black"
    ) +
    # Remove padding around coordinates so aspect ratio and top baseline align exactly
    scale_x_continuous(expand = c(0, 0)) +
    scale_y_reverse(expand = c(0, 0)) +
    scale_color_gradientn(
      colors = viridisLite::plasma(256),
      limits = c(0, scale.clamp),
      oob = scales::squish,
      values = scales::rescale(c(0, threshold_val, scale.clamp)),
      guide = "none"
    ) +
    scale_fill_manual(
      name = NULL,
      values = c("Not Diffuse Glioma" = "black"),
      guide = guide_legend(label.position = "right")
    ) +
    coord_fixed() +
    theme_void() +
    theme(
      legend.position = "bottom",
      legend.key.size = unit(0.9, "cm"),
      legend.text = element_text(size = 27, color = "black"),
      legend.margin = margin(t = 12, b = 4),
      panel.border = element_rect(colour = "grey90", fill = NA, linewidth = 1),
      plot.margin = margin(t = 0, b = 0, l = 5, r = 5)
    )
  
  # Background raster spans up to y_axis_max
  df_bar <- data.frame(y = seq(0, y_axis_max, length.out = 400), x = 1)
  
  p.risk.legend <- ggplot(df_bar, aes(x = x, y = y, fill = y)) +
    geom_raster(interpolate = TRUE) +
    scale_fill_gradientn(
      colors = viridisLite::plasma(256),
      limits = c(0, scale.clamp),
      oob = scales::squish,
      values = scales::rescale(c(0, threshold_val, scale.clamp)),
      guide = "none"
    ) +
    geom_hline(yintercept = threshold_val, color = "white", linewidth = 1.0, linetype = "dashed") +
    scale_shape_manual(values = c(1, 95)) +
    scale_size_manual(values = c(4, 16)) +
    scale_color_manual(values = c("black", "red")) +
    geom_point(data = df_dots, aes(x = x, y = y, shape = type, size = type, color = type), inherit.aes = FALSE) +
    
    annotate("text", x = 1, y = -0.10 * scale.clamp, label = "Risk score", hjust = 0.5, size = 9.6, fontface = "bold", color = "black") +
    scale_y_continuous(
      breaks = all_breaks,
      labels = all_labels,
      position = "right",
      expand = c(0, 0)
    ) +
    scale_x_continuous(limits = c(0.5, 1.5), expand = c(0, 0)) +
    coord_cartesian(ylim = c(-0.18 * scale.clamp, y_axis_max), clip = "off") +
    theme_void() +
    theme(
      legend.position = "bottom",
      legend.title = element_blank(),
      legend.text = element_text(size = 27, color = "black"),
      legend.key.size = unit(0.9, "cm"),
      legend.margin = margin(t = 12, b = 4),
      axis.text.y.right = element_text(size = 24, color = "black", hjust = 0, margin = margin(l = 8)),
      axis.ticks.y.right = element_line(color = "black", linewidth = 0.8),
      axis.ticks.length = unit(6, "pt"),
      plot.margin = margin(t = 0, b = 0, l = 10, r = 180)
    )
  
  # Calculate aspect ratio of data bounding box (width / height)
  range_x <- max(tile.merge$x2) - min(tile.merge$x1)
  range_y <- max(tile.merge$y2) - min(tile.merge$y1)
  aspect_ratio <- range_x / range_y
  
  # Fixed panel height: 600 px; Fixed scale bar width: 90 px; Floating heatmap width: 600 * aspect_ratio
  fixed_panel_h <- 600
  fixed_scale_w <- 90
  floating_heat_w <- round(fixed_panel_h * aspect_ratio)
  
  p.risk.spatial <- (p.risk.heatmap + p.risk.legend) +
    patchwork::plot_layout(widths = c(floating_heat_w, fixed_scale_w)) +
    patchwork::plot_annotation(
      title = glue("Spatial Heterogeneity of Glioma Risk ({glioma.subtype})"),
      theme = theme(
        plot.title = element_text(size = 28, face = "bold", hjust = 0.5, margin = margin(t = 15, b = 25))
      )
    )
  
  # Canvas dimensions: fixed height of 860 px to comfortably fit 3x larger text, floating width
  canvas_h <- 860
  canvas_w <- floating_heat_w + fixed_scale_w + 360
  
}else{
  p.risk.spatial<-ggplot()
  canvas_h <- 600
  canvas_w <- 600
}

######################################


png(filename=file.path(dir.base, "risk", study, sample.name, glue("{sample.name}_thumbnail.jpeg")))
print(p.thumb)
dev.off()


png(filename=file.path(dir.base, "risk", study, sample.name, glue("{sample.name}_risk_spatial.jpeg")),
    width=canvas_w, height=canvas_h)
print(p.risk.spatial)
dev.off()

pdf(file.path(dir.base, "risk", study, sample.name, glue("{sample.name}_risk_spatial.pdf")),
    width=canvas_w / 72, height=canvas_h / 72)
print(p.risk.spatial)
dev.off()


png(filename=file.path(dir.base, "risk", study, sample.name, glue("{sample.name}_histo_class.jpeg")))
print(p.class)
dev.off()

pdf(file.path(dir.base, "risk", study, sample.name, glue("{sample.name}_histo_class.pdf")))
print(p.class)
dev.off()

png(filename=file.path(dir.base, "risk", study, sample.name, glue("{sample.name}_regions.jpeg")))
print(p.heatmap)
dev.off()

pdf(file.path(dir.base, "risk", study, sample.name, glue("{sample.name}_regions.pdf")))
print(p.heatmap)
dev.off()

#################################






















