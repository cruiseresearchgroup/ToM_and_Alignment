#
#$ -j y
#$ -e $JOB_ID_$JOB_NAME.out
#$ -o $JOB_ID_$JOB_NAME.out
#$ -cwd
#$ -N latentqa
#$ -pe smp 64
#$ -l mem=100G,jobfs=180G,ngpus=1,gpu_model=H100_NVL
#$ -V
#$ -P CRUISE


export RANK=0
export WORLD_SIZE=1
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=12348
export RAYON_NUM_THREADS=48
export HF_HOME=/srv/scratch/CRUISE/Mehdi/HF


/srv/scratch/CRUISE/z5517269/miniconda/envs/latentqa/bin/python reading.py \
    --decoder_model_name /srv/scratch/CRUISE/Mehdi/out/runs/003/checkpoints/epoch4-steps4500-2025-01-01_12-11-00 \
    --target_model_name meta-llama/Llama-3.2-3B-Instruct \
    --prompt "For Agent 1: The priority for Food, Water and Firewood are respectively Low, Medium and High. For Agent 2: The priority for Food, Water and Firewood are respectively Low, Medium and High." \
    --save_name similar_priorities