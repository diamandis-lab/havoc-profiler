library(docopt)

doc<-"
Description:
Loads dlfvs, filters by no.tiles, and for each feature calculates mean per region

Usage:
amalgamate_dlfv.R --dlfv_in <FNAME> --dir_out <DIR>
amalgamate_dlfv.R --dlfv_in=<FNAME> --dir_out=<DIR>
--dlfv_in <FNAME>  i.e. /data/havoc_profiler/dlfv/vit_study_model.csv or /app_data/dlfv/... (inside docker mounts inside study)
--dir_out <DIR>  i.e. /data/havoc_profiler/dlfv_region
--help            Shows this message
"
opt <- docopt(doc)

dlfv.in<-opt$dlfv_in
dir.out<-opt$dir_out
study<-opt$study

#dlfv.in<-"/data/havoc_profiler/dlfv/vit_EBRAINS-GLIOMA_k6_px512_mpp252_kumar_prov_gigapath.csv"
#dir.out<-"/data/havoc_profiler/dlfv_region"

#opt<-list(threads=16)

library(readr)
library(dplyr)
library(reshape2)
library(glue)


features.raw<-data.frame(suppressMessages(read_csv(file=dlfv.in)))
rownames(features.raw)<-features.raw$image_path
features.raw<-features.raw[,-1]
features.raw.region<-sapply(rownames(features.raw), function(curr){
  strsplit(curr, split="/")[[1]][6]
}, USE.NAMES = FALSE)
regions.unique<-sort(unique(features.raw.region))

features.raw.region.pos<-sapply(features.raw.region, function(curr){which(regions.unique==curr)}, USE.NAMES = FALSE)

# slow with full studies
list.res<-lapply(1:length(regions.unique), function(pos){
  if(pos%%100==0){message(pos,"/",length(regions.unique))}
  features.raw.sel<-features.raw[features.raw.region.pos==pos,]
  row<-data.frame(havoc_region=regions.unique[pos],
                  Count_Tiles=nrow(features.raw.sel),
                  t(data.frame(colMeans(features.raw.sel))))
  return(row)
})

df.means<-data.frame(bind_rows(list.res))
rownames(df.means)<-NULL

write.csv(df.means, file=file.path(dir.out, gsub("vit_", "dlfv_region_", basename(dlfv.in))), row.names = FALSE)


