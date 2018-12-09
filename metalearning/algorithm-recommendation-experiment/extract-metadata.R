require(mfe)

setwd("./uci-datasets/")
datasets = system("ls ./", intern=T)

first_col_class = c(
	"segmentation.data",
	"hepatitis.data",
	"abalone.data")

custom_col_class = rbind(
	c("horse-colic.data", 24),
	c("echocardiogram.data", 2),
	c("flag.data", 7))

mtft = NULL
for (dataset in datasets) {
	cat("extracting metafeatures from", dataset, "...\n")
	cur_data = read.table(dataset, 
		fileEncoding="latin1",
		sep=",", 
		na.strings=c("NA", "?"))

	if (dataset %in% first_col_class) {
		class_index = 1
	} else if(dataset %in% custom_col_class) {
		custom_ind = which(custom_col_class[, 1] == dataset)
		class_index = as.numeric(custom_col_class[custom_ind, 2])
	} else {
		class_index = ncol(cur_data)
	}

	classes = unique(cur_data[, class_index])
	cat("class_num =", length(classes), "\n")
	for (class in classes) {
		cat("\t", class, "=", 
			sum(cur_data[, class_index] == class), "\n")
	}

	mtft = rbind(mtft, 
		metafeatures(
			cur_data[, -class_index], 
			cur_data[,  class_index], 
			groups = c("landmarking", "infotheo", "general", "model.based"))) 
}

row.names(mtft) = datasets

write.csv(mtft, 
	file="../metafeatures/metafeatures-extracted.metadata", 
	quote=F, 
	row.names=T)
