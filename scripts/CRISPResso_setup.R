args = commandArgs(trailingOnly=TRUE)
samplesheet_path <- args[1]
fastq_dir <- args[2]

samplesheet <- read.delim(samplesheet_path)

if (!("folder" %in% colnames(samplesheet))) {
    samplesheet$folder <- paste0(samplesheet$folder1, "/", samplesheet$folder2, "/")
    samplesheet$batch <- samplesheet$folder2
} else {
    samplesheet$batch <- samplesheet$folder
}
for (f in unique(samplesheet$folder)) {
	dir.create(f, recursive = TRUE, showWarnings = FALSE)
}

for (i in 1:nrow(samplesheet)) {
	if (file.exists(paste(fastq_dir,samplesheet$fastq[i], sep = "/"))) {
		file.copy(paste(fastq_dir, samplesheet$fastq[i], sep = "/")
		, paste(samplesheet$folder[i], samplesheet$fastq[i], sep = "/"))
	}
}

samplesheets <- c()
for (i in unique(samplesheet$folder)) {
	sub_samplesheet <- samplesheet[samplesheet$folder == i, c('fastq', 'g', 'a', 'w', 'Description')]
	colnames(sub_samplesheet) <- c('r1',  'g', 'a', 'w', 'n')
	write.table(sub_samplesheet, paste0(i, "/", unique(samplesheet$batch[samplesheet$folder == i]), ".txt")
	, quote = FALSE, sep = "\t", row.names = FALSE)
	samplesheets <- append(samplesheets, paste0(i, "/", unique(samplesheet$batch[samplesheet$folder == i]), ".txt"))
}

writeLines(samplesheets, "CRISPResso_batch_files.txt")
