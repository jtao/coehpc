#!/bin/bash
#SBATCH --export=NONE               
#SBATCH --get-user-env=L             
#SBATCH --job-name=tl_R
#SBATCH --output=tl_R.%j
#SBATCH --time=01:00:00            
#SBATCH --ntasks=56          
#SBATCH --ntasks-per-node=28         
#SBATCH --cpus-per-task=1
#SBATCH --mem=4096M                  
      
ml R
cd /home/jtao/scratch/BYOC/tamulauncher
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
tamulauncher commands.in
