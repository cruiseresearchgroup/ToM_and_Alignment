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


/srv/scratch/CRUISE/z5517269/miniconda/envs/latentqa/bin/python control.py \
    --decoder_model_name /srv/scratch/CRUISE/Mehdi/out/runs/028/checkpoints/epoch2-steps4005-2025-01-07_12-50-38 \
    --control similar_priorities \
    --dataset dolly \
    --eval_prompts default \
    --samples 30 

# --decoder_model_name /srv/scratch/CRUISE/Mehdi/out/runs/024/checkpoints/epoch3-steps5340-2025-01-03_22-23-33 \