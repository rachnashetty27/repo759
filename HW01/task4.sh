touch task4.sh
nano task4.sh
#!/bin/bash

#SBATCH --cpus-per-task=2     # Request 2 CPU cores
#SBATCH --job-name=FirstSlurm # Set job name to FirstSlurm
#SBATCH --output=FirstSlurm.out   # Specify output file
#SBATCH --error=FirstSlurm.err    # Specify error file

# Print the hostname of the compute node
Hostname

chmod +x task4.sh
sbatch task4.sh

squeue -u $USER
cat FirstSlurm.out
