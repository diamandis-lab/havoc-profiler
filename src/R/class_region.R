library(readr)
library(data.table)
library(dplyr)
library(parallel)


library(docopt)

"class.region: amalgamates region-level classification result to a single study-level file

Usage:
  class_region.R --study <NAME> --model_name <NAME> [--threads <NUM>]
  class_region.R --study=<NAME> --model_name=<NAME> [--threads=<NUM>]

Options:
  --study NAME i.e. TCGA-LGG_k6_px512_mpp252_kumar
  --model_name NAME i.e. glial_histology
  --threads NUM Number of threads [default: 16]
  --help               show this help text
" -> doc
opt <- docopt(doc)

threads<-as.integer(opt$threads)

study<-opt$study
model.name<-opt$model_name


dir.base <- file.path("/app_data")
dir.classifier <- file.path(dir.base, "classifier")
dir.class.region <- file.path(dir.base, "classifier_region")
if(!dir.exists(dir.class.region)){dir.create(dir.class.region)}



fnames.infer<-list.files(dir.classifier)
samples<-gsub("infer_", "", gsub(paste0("_", model.name, ".csv"), "", fnames.infer))
slides<-sort(unique(as.character(sapply(samples, function(sample){
  pos.char<-which(strsplit(sample, split="")[[1]]=="_")
  pos.char<-max(pos.char) # if multiple "_", keeps position of the latest
  slide<-substr(sample, 1, pos.char - 1) # removes region identifier "_blue"
  return(slide)
}))))

cl<-makeCluster(threads)
clusterEvalQ(cl, expr={
  library(data.table)
  library(readr)
})
clusterExport(cl, c("dir.base", "dir.classifier", "dir.class.region", "study", "model.name","slides"))


list.res<-parLapply(cl, fnames.infer, function(fname){
  pred<-read_csv(file.path(dir.classifier, fname), show_col_types = FALSE)
  row<-data.table(t(data.frame(colMeans(pred[,-1]))))
  region<-gsub("infer_", "", gsub(paste0("_", model.name, ".csv"), "", fname))
  
  vec<-t(row)[,1]
  pos.top<-which(vec==max(vec))
  top.class<-names(pos.top)[1]
  top.pred<-as.numeric(vec[pos.top])
  
  #if(top.class=="Glial histology" & top.pred<0.6666){top.class<-paste0(top.class, "-low")}
  
  row<-cbind(region=region, top_class_name=top.class, top_class_score=top.pred, row)
  
  
  return(row)
})
pred<-bind_rows(list.res)


list.havoc<-parLapply(cl, 1:length(slides), function(pos){
  slide<-slides[pos]
  message("loading havoc ", pos, "/", length(slides)," ", slide)
  fname.havoc<-list.files(file.path(dir.base, "havoc", "outputs", slide), pattern="cluster_info_df.csv")
  tb.havoc<-read_csv(file.path(dir.base, "havoc", "outputs", slide, fname.havoc), show_col_types = FALSE)
  regions<-unique(tb.havoc$Cluster_color_name_6)
  tile.count<-sapply(regions, function(region){
    sum(tb.havoc$Cluster_color_name_6==region)
  })
  df.count<-data.frame(region = paste0(slide, "_", regions), tile_count = tile.count)
  rownames(df.count)<-NULL
  return(df.count)
})

tile.count<-bind_rows(list.havoc)

region.tiles<-sapply(pred$region, function(curr){
  tile.count$tile_count[which(tile.count==curr)]
})
colnames(pred)
pred<-cbind(region=pred$region,
            region_tiles=region.tiles,
            pred[,-1])

write_csv(pred, file.path(dir.class.region, paste0("class_region_", study, "_", model.name,".csv")))



stopCluster(cl)

