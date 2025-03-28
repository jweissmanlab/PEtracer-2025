#usage: Rscript make_CRISResso_summary.R path_to_CRISPResso_output output_file

suppressPackageStartupMessages(library(dplyr))

args = commandArgs(trailingOnly=TRUE)
dir <- args[1]
output_file <- args[2]


summary_all <- data.frame(matrix(nrow = 0, ncol = 0))
for(directory in list.dirs(dir,recursive = TRUE,  full.names = TRUE)){
	quant_file <- paste0(directory,"/", "CRISPRessoBatch_quantification_of_editing_frequency.txt")
	if(file.exists(quant_file)) {
		dat <- read.delim(quant_file)
		dat_summary <- dat %>% group_by(Batch) %>% summarise(Correct_edit = (Reads_aligned[Amplicon == "Amplicon1"]/
		Reads_aligned_all_amplicons[Amplicon == "Amplicon1"])*100,
		Indels = ((sum(Insertions, Deletions)-sum(Insertions.and.Deletions))/
		Reads_aligned_all_amplicons[Amplicon == "Amplicon1"]) *100,
		Substitutions_amplicon = (Substitutions[Amplicon == "Amplicon1"]/
		Reads_aligned_all_amplicons[Amplicon == "Amplicon1"])*100,
		Reads_aligned_all_amplicons = unique(Reads_aligned_all_amplicons),
		Reads_aligned_reference = Reads_aligned[Amplicon == "Reference"],
		Reads_aligned_amplicon = Reads_aligned[Amplicon == "Amplicon1"])
		dat_summary$dir <- gsub(".*/", "", directory)
		summary_all <- rbind(summary_all, dat_summary)
	}
}

write.table(summary_all, output_file, quote = FALSE, sep = "\t", row.names = FALSE)
