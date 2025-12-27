#!/bin/bash

#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=Sabrina.ebert@desy.de

# create temporary folders in RAM
export SINGULARITY_TMPDIR=/dev/shm/sabebert/stmp
export SINGULARITY_CACHEDIR=/dev/shm/sabebert/scache
#mkdir -p $SINGULARITY_TMPDIR $SINGULARITY_CACHEDIR
mkdir -p /dev/shm/sabebert/stmp /dev/shm/sabebert/scache

# build container
#singularity build --fakeroot --force container.sif container.def

# execute command

configFiles="$DATA/ConfigFiles/kvsFiles/mdPos/Training3Kvs"  #contains kvstest.ini kvs.xml temps and all created ini files
#buildPath="/home/sabebert/MaMiCo/MaMiCo/build" #Path to build folder with kvs executable and config files
buildPath="/data/dust/user/sabebert/build"
#outputPath="/home/sabebert/MaMiCo/MaMiCo/build/output"
outputPath="$DATA/TrainingData/Training3Kvs"



SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID//{ } 
SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID//} }

#folders=(1186 1214 1431 1459) #(136 122 108 332 318 304 1186 1214 1431 1459 1676 1690 1704 1949 1935 1921 2194 2180 2166 1200 1445 969 941 955 724 710 696 220 206 234)
folders=(118620 121420 143120 145920 118626 121426 143126 145926 118628 121428 143128 145928) 

#folders=(1186 1214 1431 1459 1704 1676 1949 1921)
# This variable contains the current array job ID
task_id=$SLURM_ARRAY_TASK_ID
# Use the task_id to index into your array (bash arrays are 0-indexed)
#Slurm array indices typically start at 1
FOLDER=${folders[$((task_id-1))]}
echo "Processing array job $SLURM_ARRAY_TASK_ID with value: $FOLDER"


echo $buildPath/
#copy created kvstest-ini files with settings to executing folder
cp $configFiles/"kvstest_${FOLDER}.ini" $buildPath/"kvstest${FOLDER}.ini"

echo "${FOLDER}"
# Extract the sample name for the current $SLURM_ARRAY_TASK_ID
#mdpos=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $2}' $config)
cd $buildPath
export SLURM_ARRAY_TASK_ID=$FOLDER
envsubst < $configFiles/"kvs_temp_training3.xml" > $buildPath/"kvs$FOLDER.xml" #For smaller cylinder
#envsubst < $configFiles/"kvs_temp.xml" > $buildPath/"kvs$SLURM_ARRAY_TASK_ID.xml"

#cd $outputPath
singularity exec --nv --env SLURM_ARRAY_TASK_ID=$FOLDER ../container.sif /opt/conda/envs/mamicoTest/bin/python3 $buildPath/kvstest.py  #--output=${MamicoPath}/build/output #$SLURM_ARRAY_TASK_ID $SLURM_ARRAY_TASK_COUNT

ls -lrt

#while [ ! -f "outer_cells_macro_${SLURM_ARRAY_TASK_ID}.csv" ]; do
#	echo "Waiting for task $task_id to complete..."
#        sleep 10
#done

#Move output to output folder
FOLDERID=${SLURM_ARRAY_TASK_ID//{ }
FOLDERID=${FOLDERID//} }

#Create subfolder if not exists
mkdir -p "$outputPath/$FOLDERID"

#mv "inner-cells_macro_mo_${SLURM_ARRAY_TASK_ID}.csv" "$outputPath/$FOLDERID/"
#mv "outer-cells_macro_mo_${SLURM_ARRAY_TASK_ID}.csv" "$outputPath/$FOLDERID/"
mv "md_macro_vel_${SLURM_ARRAY_TASK_ID}.csv" "$outputPath/$FOLDERID/"
mv "outer_macro_vel_${SLURM_ARRAY_TASK_ID}.csv" "$outputPath/$FOLDERID/"
#mv "CheckpointSimpleMD_${SLURM_ARRAY_TASK_ID}"*   "$outputPath/$FOLDERID/"
mv "kvs${SLURM_ARRAY_TASK_ID}.xml"                "$outputPath/$FOLDERID/"
mv "kvstest${SLURM_ARRAY_TASK_ID}.ini"            "$outputPath/$FOLDERID/"

    


