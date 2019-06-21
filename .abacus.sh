echo running...
cat abacus.sh
sbatch --time 10:00:00 -J model_train --output=main.out --mail-user=cristi.mtr@gmail.com --mail-type=ALL abacus.sh

