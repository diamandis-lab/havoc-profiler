# If you don't have EBImage, install it via BiocManager:
# install.packages("BiocManager")
# BiocManager::install("EBImage")
library(EBImage)
library(parallel)

study.full<-"CPTAC-GBM_k6_px512_mpp252_kumar"

calculateBlankRatio <- function(image_path) {
  if (!file.exists(image_path)) return(NA)
  
  tryCatch({
    # 1. Load Image
    img <- readImage(image_path)
    
    # The Hematoxylin-Only Method (Color Deconvolution)
    #    Instead of using grayscale (luminance), which mixes the pink (Eosin) 
    #    and purple (Hematoxylin), target the nuclei specifically. 
    #    The "blank" area is just the absence of both.
    
    # Target the Red channel (where Hematoxylin absorbs most light)
    img_red <- channel(img, "red") 
    # Red channel < 0.8 is usually tissue; > 0.8 is usually background.
    blank_mask <- img_red < 0.85
    
    # 4. Calculate Percentage
    # sum(blank_mask) gives the count of white pixels
    # length(blank_mask) gives the total pixel count
    ratio_blank <- 1 - (sum(blank_mask) / length(blank_mask))
    
    return(ratio_blank)
    
  }, error = function(e) {
    message(paste("Error processing", image_path, ":", e$message))
    return(NA)
  })
}


# --- Execution ---
dir.base <- file.path("/data","havoc_profiler", "hovernet", study.full)

#files <- c("(13302, 16372, 13814, 16884).jpg", 
#           "(38912, 31232, 39424, 31744).jpg")
#
#for (file_name in files) {
#  full_path <- file.path(dir_base, file_name)
#  result <- calculate_extent_final(full_path)
#  cat(sprintf("Mean Extent for %s: %.4f\n", file_name, result))
#}


regions<-list.files(dir.base)

regionBlankRatio<-function(region){
  fnames<-list.files(file.path(dir.base, region, "tiles_selected"))
  if(length(fnames) > 25){
    fnames<-sample(fnames, 25)
  }
  res<-as.numeric(sapply(fnames, function(fname){
    calculateBlankRatio(file.path(dir.base, region, "tiles_selected", fname))
  }))
  mean.extent<-mean(res)
  return(mean.extent)
}


cl<-makeCluster(16)
r<-clusterEvalQ(cl, expr={
  library(EBImage)
})

clusterExport(cl, c("dir.base",
                    "calculateBlankRatio",
                    "regionBlankRatio"))

res<-parSapply(cl, regions, function(region){
  try({
    regionBlankRatio(region=region)
  })
})
stopCluster(cl)

df<-data.frame(REGION=names(res), RATIO_BLANK=res)
rownames(df)<-NULL

write.csv(df, file=file.path("/data","havoc_profiler", paste0("RATIO_BLANK_", study.full, ".csv")), row.names=FALSE, quote=FALSE)



