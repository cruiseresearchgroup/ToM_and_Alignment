#
#$ -j y
#$ -e $JOB_ID_$JOB_NAME.out
#$ -o $JOB_ID_$JOB_NAME.out
#$ -cwd
#$ -N latentqa
#$ -pe smp 64
#$ -l mem=200G,jobfs=180G,ngpus=1,gpu_model=L40S
#$ -V
#$ -P CRUISE


export RANK=0
export WORLD_SIZE=1
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=12348
export RAYON_NUM_THREADS=48
export HF_HOME=/srv/scratch/CRUISE/Mehdi/HF


/srv/scratch/CRUISE/z5517269/miniconda/envs/latentqa/bin/python train.py \
--target_model_name meta-llama/Meta-Llama-3-8B-Instruct \
--train_qa data/CRAIGSLISTBARGAIN/train.json \
--gradient_accumulation_steps 8 \ 
--use_wandb \
--eval_ppl \
--eval_qa data/CRAIGSLISTBARGAIN/valid.json \
--eval_every_n_steps 500