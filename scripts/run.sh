#
#$ -j y
#$ -e $JOB_ID_$JOB_NAME.out
#$ -o $JOB_ID_$JOB_NAME.out
#$ -cwd
#$ -N latentqa
#$ -pe smp 64
#$ -l mem=100G,jobfs=180G,ngpus=2,gpu_model=L40S
#$ -V
#$ -P CRUISE


export RANK=0
export WORLD_SIZE=1
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=12348
export RAYON_NUM_THREADS=48
export HF_HOME=/srv/scratch/CRUISE/Mehdi/HF


/srv/scratch/CRUISE/z5517269/miniconda/envs/latentqa/bin/python reading_ToM.py \
    --target_model_name meta-llama/Llama-3.2-3B-Instruct \
    --decoder_model_name /srv/scratch/CRUISE/Mehdi/out/runs/034/checkpoints/epoch2-steps21669-2024-12-30_15-19-54