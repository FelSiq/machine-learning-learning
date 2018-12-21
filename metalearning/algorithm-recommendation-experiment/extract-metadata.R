require(mfe)

setwd("./openml-data")
datasets = system("ls ./*.csv", intern=T)

max_rows = 5000

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
        header=T,
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

    if (nrow(cur_data) > max_rows) {
        inst_prob = cur_data[,class_index] + 1
        inst_prob = table(inst_prob)[inst_prob]
        inst_prob = inst_prob / sum(inst_prob)

        sampled_inst = sample(1:nrow(cur_data), size=max_rows, prob=inst_prob)
        cur_data = cur_data[sampled_inst,]

        rm(sampled_inst, inst_prob)
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
			groups = c("landmarking", "statistical", "infotheo", "general", "model.based"))) 

    row.names(mtft) = datasets[1:nrow(mtft)]

    write.csv(mtft, 
        file="../metafeatures/metaf-extracted-openml.metadata", 
        quote=F, 
        row.names=T)
}
