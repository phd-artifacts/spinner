# Using with SLURM

Spinner is designed to work well with Slurm, but for simplicity it does not directly launch or interact with jobs.

```sh
#!/bin/bash
#SBATCH --nodes=4
#SBATCH --partition=gpu-partition
#SBATCH --time=04:00:00
#SBATCH --job-name=cool-benchmark
#SBATCH --output=job_log_%j.out

set -e

# Load required modules
module purge
module load cmake/3.27.9
module load mpich/4.0.2
module load singularity/3.7.1

# Load Spinner env
source spinner/.venv/bin/activate

# Get MPI host lsit from Slurm and format it for MPI
host_list=$(scontrol show hostname $(echo "$SLURM_JOB_NODELIST" | head -n 4 | tr '\n' ',' | sed 's/,$//'))

spinner -c bench_settings.yaml -r -e -o mpi_bench.pkl --extra-args "hosts=$host_list"

```