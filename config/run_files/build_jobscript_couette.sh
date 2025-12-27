#!/bin/bash

#SBATCH --job-name=couette_array
#SBATCH --output=couette_%A_%a.out
#SBATCH --error=couette_%A_%a.err
#SBATCH --time=3-24:00:00
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=Sabrina.ebert@desy.de
#SBATCH --array=1-21%10  # 21 couette files found, max 10 concurrent jobs

# create temporary folders in RAM
export SINGULARITY_TMPDIR=/dev/shm/sabebert/stmp
export SINGULARITY_CACHEDIR=/dev/shm/sabebert/scache
mkdir -p /dev/shm/sabebert/stmp /dev/shm/sabebert/scache

# Set paths
WORKSPACE_DIR="/data/dust/user/sabebert/build" #"/home/sabrina/Desktop/Schreibtisch/Masterarbeit/Github/MaMiCo_hybrid_ml/0_Data_Generation
COUETTE_DIR="$WORKSPACE_DIR/0_couette"
CONTAINER_PATH="$DATA/container.sif"
# Create a file list if it doesn't exist
COUETTE_FILES_LIST="$WORKSPACE_DIR/couette_files_list.txt"

# Function to find all couette files and create the list
create_couette_files_list() {
    echo "Scanning for couette files in $COUETTE_DIR..."
    find "$COUETTE_DIR" -name "couette" -type f > "$COUETTE_FILES_LIST"
    echo "Found $(wc -l < "$COUETTE_FILES_LIST") couette files"
}

# Create the file list if it doesn't exist
if [ ! -f "$COUETTE_FILES_LIST" ]; then
    create_couette_files_list
fi

# Get the total number of couette files
TOTAL_FILES=$(wc -l < "$COUETTE_FILES_LIST")
echo $TOTAL_FILES
# Check if SLURM_ARRAY_TASK_ID is within valid range
if [ "$SLURM_ARRAY_TASK_ID" -gt "$TOTAL_FILES" ]; then
    echo "SLURM_ARRAY_TASK_ID ($SLURM_ARRAY_TASK_ID) exceeds total number of files ($TOTAL_FILES)"
    exit 0
fi

# Get the couette file path for this array task
COUETTE_FILE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$COUETTE_FILES_LIST")
COUETTE_DIR_PATH=$(dirname "$COUETTE_FILE")

echo "Processing array job $SLURM_ARRAY_TASK_ID"
echo "Couette file: $COUETTE_FILE"
echo "Working directory: $COUETTE_DIR_PATH"

# Change to the directory containing the couette file
cd "$COUETTE_DIR_PATH"

# Check if container exists
if [ ! -f "$CONTAINER_PATH" ]; then
    echo "Error: Container not found at $CONTAINER_PATH"
    exit 1
fi

# Check if couette file exists and is executable
if [ ! -f "$COUETTE_FILE" ]; then
    echo "Error: Couette file not found at $COUETTE_FILE"
    exit 1
fi

if [ ! -x "$COUETTE_FILE" ]; then
    echo "Making couette file executable..."
    chmod +x "$COUETTE_FILE"
fi

# Check if couette.xml exists in the same directory
if [ ! -f "couette.xml" ]; then
    echo "Warning: couette.xml not found in $COUETTE_DIR_PATH"
fi

# Execute the couette simulation
echo "Starting couette simulation..."
singularity exec "$CONTAINER_PATH" "$COUETTE_FILE"

echo "Couette simulation completed for task $SLURM_ARRAY_TASK_ID" 
