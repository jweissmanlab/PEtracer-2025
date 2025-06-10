Rscript ../scripts/CRISPResso_setup.R CRISPResso_samplesheet.txt fastq/

while read -r line
do
f="$(basename -- $line)"
dir="$(dirname -- $line)"
CWD=$PWD
cd $dir
sbatch --job-name="${f%.txt}"_crispresso --nodes=1 --ntasks=1 --cpus-per-task=20 --mem=50gb --partition=20 --wrap "CRISPRessoBatch --batch_settings $f --min_average_read_quality 30 "
cd $CWD
done < CRISPResso_batch_files.txt

