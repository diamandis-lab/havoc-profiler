library(docopt)
library(readr)
library(xgboost)
library(reshape2)
suppressMessages(library(data.table))
suppressMessages(library(dplyr))
library(parallel)
library(jsonlite)
library(glue)

"infer_region_class.R: takes morphometric features as input for risk inference with an XGBoost model
Usage:
  infer_region_class.R --study <NAME> --vit_model <NAME> --classifier_model <NAME> [--threads <NUM>]
  infer_region_class.R --study=<NAME> --vit_model <NAME> --classifier_model=<NAME> [--threads=<NUM>]

Options:
  --study NAME i.e. TCGA-LGG_k6_px512_mpp252_kumar
  --vit_model NAME i.e. prov_gigapath
  --classifier_model NAME i.e. glial_histology_v3
  --threads NUM Number of threads [default: 16]
  --help               show this help text
" -> doc
opt <- docopt(doc)

study<-opt$study
vit.model<-opt$vit_model
classifier.model<-opt$classifier_model
threads<-as.numeric(opt$threads)

dir.havoc <- file.path("/data", "havoc_profiler","havoc", study)
#dir.havoc <- file.path("/data", "havoc_profiler_study",study,"havoc")

dir.hovernet <- file.path("/data", "havoc_profiler","hovernet", study)
#dir.hovernet <- file.path("/data", "havoc_profiler_study",study,"hovernet")

dir.classifier_region<-"/data/havoc_profiler/classifier_region"
dir.dlfv<-"/data/havoc_profiler/dlfv"
dir.model<-file.path("/app/model_XGBoost")

xgb<-xgb.load(file.path(dir.model, glue("xgb_{classifier.model}.model")))
df.classes<-fromJSON(file.path(dir.model, glue("xgb_{classifier.model}.json")))

#study<-"test_k6_px512_mpp252_kumar"
#classifier.model<-"glial_histology_v2"
# opt<-list(study="test_k6_px512_mpp252_kumar", classifier_model="glial_histology_v2", threads=16)



vit <- read_csv(file.path(dir.dlfv, glue("vit_{study}_{vit.model}.csv")), show_col_types = FALSE)


parseData<-function(features, dir_base, is.train=TRUE){
  # dir_base is used to extract class name, ie: "data/glial_histology_v3-train/"
  features$image_path<-gsub(dir_base, "", features$image_path)
  if(is.train){
    class<-colsplit(features$image_path, "/", letters[1:10])$a
    class.names<-sort(unique(class))
    class.id<-0:(length(class.names)-1)
    names(class.id)<-class.names
    label.curr<-as.numeric(sapply(class, function(curr) {
      class.id[names(class.id)==curr]
    }))
  }else{
    label.curr=NULL
    class.id=NULL
  }
  return(list(data=as.matrix(features[,-1]), tile_name=features$image_path, label=label.curr, class.id=class.id))
}

list.tiles<-parseData(features=vit, dir_base=dir.hovernet, is.train=FALSE)

pred<-data.frame(predict(xgb, list.tiles$data, reshape=TRUE))
colnames(pred)<-df.classes$class_name
rownames(pred)<-as.character(sapply(list.tiles$tile_name, function(curr){substr(curr, 2, nchar(curr))})) # removes first slash char: /VAL207_purple/tiles_selected/(19968, 27648, 20480, 28160).jpg

vec.region<-colsplit(rownames(pred), "/", letters[1:4])$a
regions<-sort(unique(vec.region))

list.res<-lapply(regions, function(region.curr){
  #region.curr<-regions[1]
  pred.sel<-pred[vec.region==region.curr,]
  row<-data.table(t(data.frame(colMeans(pred.sel))))
  
  vec<-t(row)
  pos.top<-which(vec==max(vec))
  top.class<-colnames(row)[pos.top]
  top.pred<-as.numeric(vec[pos.top])
  row<-cbind(region=region.curr, region_tiles=-1, top_class_name=top.class, top_class_score=top.pred, row)
  return(row)
})
res<-bind_rows(list.res)

slides<-sort(unique(as.character(sapply(regions, function(region){
  pos.char<-which(strsplit(region, split="")[[1]]=="_")
  pos.char<-max(pos.char) # if multiple "_", keeps position of the latest
  slide<-substr(region, 1, pos.char - 1) # removes region identifier "_blue"
  return(slide)
}))))

cl<-makeCluster(threads)
r<-clusterEvalQ(cl, expr={
  library(data.table)
  library(readr)
})
clusterExport(cl, c("dir.havoc","slides"))

list.havoc<-parLapply(cl, 1:length(slides), function(pos){
#list.havoc<-lapply(1:length(slides), function(pos){
  slide<-slides[pos]
  message("loading havoc ", pos, "/", length(slides)," ", slide)
  fname.havoc<-list.files(file.path(dir.havoc, slide), pattern="cluster_info_df.csv")
  tb.havoc<-read_csv(file.path(dir.havoc, slide, fname.havoc), show_col_types = FALSE)
  regions<-unique(tb.havoc$Cluster_color_name_6)
  tile.count<-sapply(regions, function(region){
    sum(tb.havoc$Cluster_color_name_6==region)
  })
  df.count<-data.frame(region = paste0(slide, "_", regions), tile_count = tile.count)
  rownames(df.count)<-NULL
  return(df.count)
})
stopCluster(cl)

tile.count<-bind_rows(list.havoc)
region.tiles<-sapply(res$region, function(curr){
  tile.count$tile_count[which(tile.count==curr)]
})

res$region_tiles<-region.tiles

write_csv(res, file.path(dir.classifier_region, glue("class_region_{study}_{classifier.model}.csv")))

message("\U2705 Saved file: ", file.path(dir.classifier_region, glue("class_region_{study}_{classifier.model}.csv")))



