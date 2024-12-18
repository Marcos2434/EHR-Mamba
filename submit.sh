#!/bin/bash
#BSUB -q hpc             # Specify the queue
#BSUB -J mamba_training_1  # Job name
#BSUB -n 4               # Number of cores
#BSUB -R "rusage[mem=8GB]"  # Memory per core
#BSUB -W 48:00           # Walltime limit (hh:mm)
#BSUB -o output_%J.out   # Standard output file
#BSUB -e error_%J.err    # Standard error file

# Load necessary modules
module load python3/3.9.19 # Load the appropriate Python module
module load cuda/11.8
module load cudnn/v8.8.0-prod-cuda-11.X
# Load other modules if necessary
pip install -r requirements.txt
pip install torch_scatter --extra-index-url https://data.pyg.org/whl/torch-2.2.0+cu118.html

# Activate virtual environment if applicable
# source /path/to/venv/bin/activate

# Navigate to the script directory
cd /dtu/blackhole/15/168981/EHR-Mamba

# Run your Python script with the desired arguments
python cli.py --output_path=output-mamba-181431 --model_type=simple_mamba --epochs=120 --batch_size=32 --lr=0.001