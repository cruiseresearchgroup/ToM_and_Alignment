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


/srv/scratch/CRUISE/z5517269/miniconda/envs/latentqa/bin/python control.py \
    --decoder_model_name /srv/scratch/CRUISE/Mehdi/out/runs/023/checkpoints/epoch2-steps4005-2025-01-03_13-18-30 \
    --control similar_priorities \
    --dataset dolly \
    --eval_prompts default \
    --samples 30 