require(mfe)

setwd("./openml-data")
datasets = system("ls ./*.csv", intern=T)

MAX_ROWS = 5000
STARTS_FROM = 20

first_col_class = c(
	"segmentation.data",
 	"hepatitis.data",
 	"abalone.data")
 
custom_col_class = rbind(
 	c("horse-colic.data", 24),
 	c("echocardiogram.data", 2),
 	c("flag.data", 7))

mtft = NULL
errors = NULL
accepted_datasets = NULL

for (dataset in datasets[STARTS_FROM:length(datasets)]) {
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

    if (MAX_ROWS > 0 && nrow(cur_data) > MAX_ROWS) {
        inst_prob = cur_data[,class_index] + 1
        table_aux = table(inst_prob)
        table_aux = table_aux / sum(table_aux)
        inst_prob = table_aux[inst_prob]
        inst_prob = inst_prob / sum(inst_prob)

        sampled_inst = sample(1:nrow(cur_data), size=MAX_ROWS, prob=inst_prob)
        cur_data = cur_data[sampled_inst,]

        rm(sampled_inst, inst_prob)
    }

	classes = unique(cur_data[, class_index])
	cat("class_num =", length(classes), "\n")
	for (class in classes) {
		cat("\t", class, "=", 
			sum(cur_data[, class_index] == class), "\n")
	}

    aux = tryCatch (
            metafeatures(
                    cur_data[, -class_index], 
                    cur_data[,  class_index], 
                    groups = c("landmarking", "infotheo", "general", "model.based")),
            error=function(error_message) {
                cat("Failed at", dataset, "dataset. \n")
                errors <<- c(errors, dataset)
                return (NULL)
            }
    )

    if (!is.null(aux)) {
            accepted_datasets <- c(accepted_datasets, dataset)

            mtft = rbind(mtft, aux)

            row.names(mtft) = accepted_datasets

            write.csv(mtft, 
                file="../metafeatures/metaf-extracted-openml3.metadata", 
                quote=F, 
                row.names=T)
    }
}

cat("\nFinished with total of", length(accepted_datasets),
    " processed datasets and", length(errors), "errors:\n")
print(errors)
