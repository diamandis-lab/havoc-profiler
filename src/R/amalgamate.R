library(docopt)

doc<-"
Description:
Loads features Image.csv files, selects \"useful\" features, filters by no.tiles and no.nuclobj, and for each feature calculates median per region

Usage:
amalgamate.R --dir_in <DIR> --dir_out <DIR> --study <NAME> [--threads <NUM>]
amalgamate.R --dir_in=<DIR> --dir_out=<DIR> --study=<NAME> [--threads=<NUM>]
--dir_in <DIR>  i.e. /data/havoc_profiler/cellprofiler/TCGA-GBM_k6_px512_mpp252 or /app_data/cellprofiler (inside docker mounts inside study)
--dir_out <DIR>  i.e. /data/havoc_profiler/features
--study <NAME> i.e. TCGA-GBM_k6_px512_mpp252
--threads <NUM>   Number of parallel execution threads [default: 16]
--help            Shows this message
"
opt <- docopt(doc)

dir.in<-opt$dir_in
dir.out<-opt$dir_out
study<-opt$study

#dir.in<-"/data2/havoc_profiler_study/GBM_havoc_paper_k6_px512_mpp252_kumar/cellprofiler"
#dir.out<-"/data/havoc_profiler/features"
#study<-"GBM_havoc_paper"
#opt<-list(threads=16)

library(readr)
library(dplyr)
library(reshape2)
library(parallel)

cutoff.no.tiles<-10
cutoff.no.nucobj<-1

cl <- makeCluster(as.numeric(opt$threads))

feature.group<-c("AreaOccupied","Count","Granularity","Intensity","Texture","Mean-NucleiObjects", "StDev-NucleiObjects")
feature.exclude<-c("Intensity_PercentMaximal_Grayscale") # mostly NA values


clusterExport(cl, c("study", "dir.in"))
clusterEvalQ(cl, {
  library(readr)
})


samples<-list.files(dir.in)
# excludes folders that do not contain Image.csv
is.file<-sapply(samples, function(sample){
  file.exists(file.path(dir.in, sample, "features", "Image.csv"))
})
samples<-samples[is.file]
dat<-parLapply(cl, samples, function(sample){
  fname<-file.path(dir.in, sample, "features", "Image.csv")
  tb<-read_csv(fname, show_col_types = FALSE)
  tb<-cbind(study=study, sample=sample, tb)
})
names(dat)<-samples

list.tiles<-list()
list.means<-vector(mode="list", length=length(dat))
for(list.pos in 1:length(dat)){
  d<-dat[[list.pos]]
  d<-d[d$Count_NucleiObjects>=cutoff.no.nucobj,] # removes tiles with less than 40 NucleiObjects
  if(!is.null(d)){
    if(nrow(d)>=cutoff.no.tiles){ # excludes samples with less than 100 tiles
      tile.id<-d$FileName_Origin
      colnames(d)<-gsub("Mean_NucleiObjects","Mean-NucleiObjects", colnames(d)) # renaming allows the 'colsplit' strategy below
      colnames(d)<-gsub("StDev_NucleiObjects","StDev-NucleiObjects", colnames(d)) # renaming allows the 'colsplit' strategy below
      d<-d[,sapply(colsplit(colnames(d),"_", letters[1:8])$a, function(curr) curr %in% feature.group)]
      #is.constant<-apply(d, 2, function(x) IQR(x, na.rm=TRUE))==0
      #d<-d[,!is.constant]
      row<-data.frame(matrix(apply(d, 2, function(x) median(x, na.rm=TRUE)), nrow=1))
      colnames(row)<-colnames(d)
      row<-cbind(Count_Tiles=nrow(d), row)
      rownames(row)<-names(dat)[list.pos]
      row<-row[,!colnames(row) %in% feature.exclude]
      colnames(row)<-gsub("Mean-NucleiObjects","Mean_NucleiObjects", colnames(row)) # restores original feature name
      list.means[[list.pos]]<-row
      colnames(d)<-gsub("Mean-NucleiObjects","Mean_NucleiObjects", colnames(d)) # restores original feature name
      
      chunks<-strsplit(names(dat)[list.pos], "_")[[1]]
      SLIDE_ID=paste0(chunks[1:length(chunks)-1], collapse="_")
      chunks2<-strsplit(SLIDE_ID, "-")[[1]]
      PATIENT_ID=paste0(chunks2[1:3], collapse="-")
      list.tiles[[names(dat)[list.pos]]]<-cbind(PATIENT_ID=PATIENT_ID, 
                                                HAVOC_REGION=names(dat)[list.pos], 
                                                TILE_ID=tile.id,
                                                d)
    }
  }else{
    # empty data block; null value in 'list.means' excluded later during 'bind_rows'
  }
}
df.means<-bind_rows(list.means)
df.means<-cbind(havoc_region=rownames(df.means), df.means)
rownames(df.means)<-NULL

write.csv(df.means, file=file.path(dir.out, paste0("features_",study,".csv")), row.names = FALSE)

df.tiles<-bind_rows(list.tiles)
saveRDS(df.tiles, file=file.path(dir.out, paste0("features-tile_",study,".rds")))


