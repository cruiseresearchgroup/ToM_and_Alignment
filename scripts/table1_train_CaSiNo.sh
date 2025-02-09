#
#$ -j y
#$ -e $JOB_ID_$JOB_NAME.out
#$ -o $JOB_ID_$JOB_NAME.out
#$ -cwd
#$ -N latentqa
#$ -pe smp 64
#$ -l mem=200G,jobfs=180G,ngpus=1,gpu_model=H100_NVL
#$ -V
#$ -P CRUISE

# Configuration file path
config_file_path="./lit/configs/train_config.py"

export RANK=0
export WORLD_SIZE=1
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=12348
export RAYON_NUM_THREADS=48
export HF_HOME=/srv/scratch/CRUISE/Mehdi/HF


################################################################################
################################### Llama3 1B ##################################
################################################################################


# Parameters to change
new_run_name="CaSiNo Middle Llama-3 1B"
new_target_model_name="meta-llama/Llama-3.2-1B-Instruct"
new_eval_every_n_steps=300
new_train_qa="./data/CaSiNo/train.json"
new_eval_qa="./data/CaSiNo/valid.json"
new_min_layer_to_read=8
new_max_layer_to_read=9

# Update the configuration file
sed -i "s/^.*run_name: str = .*/    run_name: str = \"$new_run_name\"/" "$config_file_path"
sed -i "s/^.*target_model_name: str = .*/    target_model_name: str = \"$new_target_model_name\"/" "$config_file_path"
sed -i "s/^.*eval_every_n_steps: int = .*/    eval_every_n_steps: int = $new_eval_every_n_steps/" "$config_file_path"
sed -i "s|^.*eval_qa: str = .*|    eval_qa: str = \"$new_eval_qa\"|" "$config_file_path"
sed -i "s/^.*min_layer_to_read: int = .*/    min_layer_to_read: int = $new_min_layer_to_read/" "$config_file_path"
sed -i "s/^.*max_layer_to_read: int = .*/    max_layer_to_read: int = $new_max_layer_to_read/" "$config_file_path"

# Running the actual experiment
/srv/scratch/CRUISE/z5517269/miniconda/envs/latentqa/bin/python train.py \
--target_model_name $new_target_model_name \
--train_qa $new_train_qa \
--gradient_accumulation_steps 8 \ 
--use_wandb \
--eval_ppl \
--eval_qa $new_eval_qa \
--eval_every_n_steps $new_eval_every_n_steps


# Parameters to change
new_run_name="CaSiNo Shallow Llama-3 1B"
new_target_model_name="meta-llama/Llama-3.2-1B-Instruct"
new_eval_every_n_steps=300
new_train_qa="./data/CaSiNo/train.json"
new_eval_qa="./data/CaSiNo/valid.json"
new_min_layer_to_read=3
new_max_layer_to_read=4

# Update the configuration file
sed -i "s/^.*run_name: str = .*/    run_name: str = \"$new_run_name\"/" "$config_file_path"
sed -i "s/^.*target_model_name: str = .*/    target_model_name: str = \"$new_target_model_name\"/" "$config_file_path"
sed -i "s/^.*eval_every_n_steps: int = .*/    eval_every_n_steps: int = $new_eval_every_n_steps/" "$config_file_path"
sed -i "s|^.*eval_qa: str = .*|    eval_qa: str = \"$new_eval_qa\"|" "$config_file_path"
sed -i "s/^.*min_layer_to_read: int = .*/    min_layer_to_read: int = $new_min_layer_to_read/" "$config_file_path"
sed -i "s/^.*max_layer_to_read: int = .*/    max_layer_to_read: int = $new_max_layer_to_read/" "$config_file_path"

# Running the actual experiment
/srv/scratch/CRUISE/z5517269/miniconda/envs/latentqa/bin/python train.py \
--target_model_name $new_target_model_name \
--train_qa $new_train_qa \
--gradient_accumulation_steps 8 \ 
--use_wandb \
--eval_ppl \
--eval_qa $new_eval_qa \
--eval_every_n_steps $new_eval_every_n_steps

# Parameters to change
new_run_name="CaSiNo Deep Llama-3 1B"
new_target_model_name="meta-llama/Llama-3.2-1B-Instruct"
new_eval_every_n_steps=300
new_train_qa="./data/CaSiNo/train.json"
new_eval_qa="./data/CaSiNo/valid.json"
new_min_layer_to_read=14
new_max_layer_to_read=15

# Update the configuration file
sed -i "s/^.*run_name: str = .*/    run_name: str = \"$new_run_name\"/" "$config_file_path"
sed -i "s/^.*target_model_name: str = .*/    target_model_name: str = \"$new_target_model_name\"/" "$config_file_path"
sed -i "s/^.*eval_every_n_steps: int = .*/    eval_every_n_steps: int = $new_eval_every_n_steps/" "$config_file_path"
sed -i "s|^.*eval_qa: str = .*|    eval_qa: str = \"$new_eval_qa\"|" "$config_file_path"
sed -i "s/^.*min_layer_to_read: int = .*/    min_layer_to_read: int = $new_min_layer_to_read/" "$config_file_path"
sed -i "s/^.*max_layer_to_read: int = .*/    max_layer_to_read: int = $new_max_layer_to_read/" "$config_file_path"

# Running the actual experiment
/srv/scratch/CRUISE/z5517269/miniconda/envs/latentqa/bin/python train.py \
--target_model_name $new_target_model_name \
--train_qa $new_train_qa \
--gradient_accumulation_steps 8 \ 
--use_wandb \
--eval_ppl \
--eval_qa $new_eval_qa \
--eval_every_n_steps $new_eval_every_n_steps

################################################################################
################################### Llama3 3B ##################################
################################################################################

# Parameters to change
new_run_name="CaSiNo Middle Llama-3 3B"
new_target_model_name="meta-llama/Llama-3.2-3B-Instruct"
new_eval_every_n_steps=300
new_train_qa="./data/CaSiNo/train.json"
new_eval_qa="./data/CaSiNo/valid.json"
new_min_layer_to_read=15
new_max_layer_to_read=16

# Update the configuration file
sed -i "s/^.*run_name: str = .*/    run_name: str = \"$new_run_name\"/" "$config_file_path"
sed -i "s/^.*target_model_name: str = .*/    target_model_name: str = \"$new_target_model_name\"/" "$config_file_path"
sed -i "s/^.*eval_every_n_steps: int = .*/    eval_every_n_steps: int = $new_eval_every_n_steps/" "$config_file_path"
sed -i "s|^.*eval_qa: str = .*|    eval_qa: str = \"$new_eval_qa\"|" "$config_file_path"
sed -i "s/^.*min_layer_to_read: int = .*/    min_layer_to_read: int = $new_min_layer_to_read/" "$config_file_path"
sed -i "s/^.*max_layer_to_read: int = .*/    max_layer_to_read: int = $new_max_layer_to_read/" "$config_file_path"

# Running the actual experiment
/srv/scratch/CRUISE/z5517269/miniconda/envs/latentqa/bin/python train.py \
--target_model_name $new_target_model_name \
--train_qa $new_train_qa \
--gradient_accumulation_steps 8 \ 
--use_wandb \
--eval_ppl \
--eval_qa $new_eval_qa \
--eval_every_n_steps $new_eval_every_n_steps


# Parameters to change
new_run_name="CaSiNo Shallow Llama-3 3B"
new_target_model_name="meta-llama/Llama-3.2-3B-Instruct"
new_eval_every_n_steps=300
new_train_qa="./data/CaSiNo/train.json"
new_eval_qa="./data/CaSiNo/valid.json"
new_min_layer_to_read=5
new_max_layer_to_read=6

# Update the configuration file
sed -i "s/^.*run_name: str = .*/    run_name: str = \"$new_run_name\"/" "$config_file_path"
sed -i "s/^.*target_model_name: str = .*/    target_model_name: str = \"$new_target_model_name\"/" "$config_file_path"
sed -i "s/^.*eval_every_n_steps: int = .*/    eval_every_n_steps: int = $new_eval_every_n_steps/" "$config_file_path"
sed -i "s|^.*eval_qa: str = .*|    eval_qa: str = \"$new_eval_qa\"|" "$config_file_path"
sed -i "s/^.*min_layer_to_read: int = .*/    min_layer_to_read: int = $new_min_layer_to_read/" "$config_file_path"
sed -i "s/^.*max_layer_to_read: int = .*/    max_layer_to_read: int = $new_max_layer_to_read/" "$config_file_path"

# Running the actual experiment
/srv/scratch/CRUISE/z5517269/miniconda/envs/latentqa/bin/python train.py \
--target_model_name $new_target_model_name \
--train_qa $new_train_qa \
--gradient_accumulation_steps 8 \ 
--use_wandb \
--eval_ppl \
--eval_qa $new_eval_qa \
--eval_every_n_steps $new_eval_every_n_steps

# Parameters to change
new_run_name="CaSiNo Deep Llama-3 3B"
new_target_model_name="meta-llama/Llama-3.2-3B-Instruct"
new_eval_every_n_steps=300
new_train_qa="./data/CaSiNo/train.json"
new_eval_qa="./data/CaSiNo/valid.json"
new_min_layer_to_read=25
new_max_layer_to_read=26

# Update the configuration file
sed -i "s/^.*run_name: str = .*/    run_name: str = \"$new_run_name\"/" "$config_file_path"
sed -i "s/^.*target_model_name: str = .*/    target_model_name: str = \"$new_target_model_name\"/" "$config_file_path"
sed -i "s/^.*eval_every_n_steps: int = .*/    eval_every_n_steps: int = $new_eval_every_n_steps/" "$config_file_path"
sed -i "s|^.*eval_qa: str = .*|    eval_qa: str = \"$new_eval_qa\"|" "$config_file_path"
sed -i "s/^.*min_layer_to_read: int = .*/    min_layer_to_read: int = $new_min_layer_to_read/" "$config_file_path"
sed -i "s/^.*max_layer_to_read: int = .*/    max_layer_to_read: int = $new_max_layer_to_read/" "$config_file_path"

# Running the actual experiment
/srv/scratch/CRUISE/z5517269/miniconda/envs/latentqa/bin/python train.py \
--target_model_name $new_target_model_name \
--train_qa $new_train_qa \
--gradient_accumulation_steps 8 \ 
--use_wandb \
--eval_ppl \
--eval_qa $new_eval_qa \
--eval_every_n_steps $new_eval_every_n_steps

################################################################################
################################### Llama3 8B ##################################
################################################################################

# Parameters to change
new_run_name="CaSiNo Middle Llama-3 8B"
new_target_model_name="meta-llama/Meta-Llama-3-8B-Instruct"
new_eval_every_n_steps=300
new_train_qa="./data/CaSiNo/train.json"
new_eval_qa="./data/CaSiNo/valid.json"
new_min_layer_to_read=15
new_max_layer_to_read=16

# Update the configuration file
sed -i "s/^.*run_name: str = .*/    run_name: str = \"$new_run_name\"/" "$config_file_path"
sed -i "s/^.*target_model_name: str = .*/    target_model_name: str = \"$new_target_model_name\"/" "$config_file_path"
sed -i "s/^.*eval_every_n_steps: int = .*/    eval_every_n_steps: int = $new_eval_every_n_steps/" "$config_file_path"
sed -i "s|^.*eval_qa: str = .*|    eval_qa: str = \"$new_eval_qa\"|" "$config_file_path"
sed -i "s/^.*min_layer_to_read: int = .*/    min_layer_to_read: int = $new_min_layer_to_read/" "$config_file_path"
sed -i "s/^.*max_layer_to_read: int = .*/    max_layer_to_read: int = $new_max_layer_to_read/" "$config_file_path"

# Running the actual experiment
/srv/scratch/CRUISE/z5517269/miniconda/envs/latentqa/bin/python train.py \
--target_model_name $new_target_model_name \
--train_qa $new_train_qa \
--gradient_accumulation_steps 8 \ 
--use_wandb \
--eval_ppl \
--eval_qa $new_eval_qa \
--eval_every_n_steps $new_eval_every_n_steps


# Parameters to change
new_run_name="CaSiNo Shallow Llama-3 8B"
new_target_model_name="meta-llama/Meta-Llama-3-8B-Instruct"
new_eval_every_n_steps=300
new_train_qa="./data/CaSiNo/train.json"
new_eval_qa="./data/CaSiNo/valid.json"
new_min_layer_to_read=5
new_max_layer_to_read=6

# Update the configuration file
sed -i "s/^.*run_name: str = .*/    run_name: str = \"$new_run_name\"/" "$config_file_path"
sed -i "s/^.*target_model_name: str = .*/    target_model_name: str = \"$new_target_model_name\"/" "$config_file_path"
sed -i "s/^.*eval_every_n_steps: int = .*/    eval_every_n_steps: int = $new_eval_every_n_steps/" "$config_file_path"
sed -i "s|^.*eval_qa: str = .*|    eval_qa: str = \"$new_eval_qa\"|" "$config_file_path"
sed -i "s/^.*min_layer_to_read: int = .*/    min_layer_to_read: int = $new_min_layer_to_read/" "$config_file_path"
sed -i "s/^.*max_layer_to_read: int = .*/    max_layer_to_read: int = $new_max_layer_to_read/" "$config_file_path"

# Running the actual experiment
/srv/scratch/CRUISE/z5517269/miniconda/envs/latentqa/bin/python train.py \
--target_model_name $new_target_model_name \
--train_qa $new_train_qa \
--gradient_accumulation_steps 8 \ 
--use_wandb \
--eval_ppl \
--eval_qa $new_eval_qa \
--eval_every_n_steps $new_eval_every_n_steps

# Parameters to change
new_run_name="CaSiNo Deep Llama-3 8B"
new_target_model_name="meta-llama/Meta-Llama-3-8B-Instruct"
new_eval_every_n_steps=300
new_train_qa="./data/CaSiNo/train.json"
new_eval_qa="./data/CaSiNo/valid.json"
new_min_layer_to_read=25
new_max_layer_to_read=26

# Update the configuration file
sed -i "s/^.*run_name: str = .*/    run_name: str = \"$new_run_name\"/" "$config_file_path"
sed -i "s/^.*target_model_name: str = .*/    target_model_name: str = \"$new_target_model_name\"/" "$config_file_path"
sed -i "s/^.*eval_every_n_steps: int = .*/    eval_every_n_steps: int = $new_eval_every_n_steps/" "$config_file_path"
sed -i "s|^.*eval_qa: str = .*|    eval_qa: str = \"$new_eval_qa\"|" "$config_file_path"
sed -i "s/^.*min_layer_to_read: int = .*/    min_layer_to_read: int = $new_min_layer_to_read/" "$config_file_path"
sed -i "s/^.*max_layer_to_read: int = .*/    max_layer_to_read: int = $new_max_layer_to_read/" "$config_file_path"

# Running the actual experiment
/srv/scratch/CRUISE/z5517269/miniconda/envs/latentqa/bin/python train.py \
--target_model_name $new_target_model_name \
--train_qa $new_train_qa \
--gradient_accumulation_steps 8 \ 
--use_wandb \
--eval_ppl \
--eval_qa $new_eval_qa \
--eval_every_n_steps $new_eval_every_n_steps


# ###############################################################################
# ################################## Ministral 8B ###############################
# ###############################################################################

# # Parameters to change
# new_run_name="CaSiNo Middle Ministral 8B"
# new_target_model_name="mistralai/Ministral-8B-Instruct-2410"
# new_eval_every_n_steps=300
# new_train_qa="./data/CaSiNo/train.json"
# new_eval_qa="./data/CaSiNo/valid.json"
# new_min_layer_to_read=15
# new_max_layer_to_read=16

# # Update the configuration file
# sed -i "s/^.*run_name: str = .*/    run_name: str = \"$new_run_name\"/" "$config_file_path"
# sed -i "s/^.*target_model_name: str = .*/    target_model_name: str = \"$new_target_model_name\"/" "$config_file_path"
# sed -i "s/^.*eval_every_n_steps: int = .*/    eval_every_n_steps: int = $new_eval_every_n_steps/" "$config_file_path"
# sed -i "s|^.*eval_qa: str = .*|    eval_qa: str = \"$new_eval_qa\"|" "$config_file_path"
# sed -i "s/^.*min_layer_to_read: int = .*/    min_layer_to_read: int = $new_min_layer_to_read/" "$config_file_path"
# sed -i "s/^.*max_layer_to_read: int = .*/    max_layer_to_read: int = $new_max_layer_to_read/" "$config_file_path"

# # Running the actual experiment
# /srv/scratch/CRUISE/z5517269/miniconda/envs/latentqa/bin/python train.py \
# --target_model_name $new_target_model_name \
# --train_qa $new_train_qa \
# --gradient_accumulation_steps 8 \ 
# --use_wandb \
# --eval_ppl \
# --eval_qa $new_eval_qa \
# --eval_every_n_steps $new_eval_every_n_steps


# # Parameters to change
# new_run_name="CaSiNo Shallow Ministral 8B"
# new_target_model_name="mistralai/Ministral-8B-Instruct-2410"
# new_eval_every_n_steps=300
# new_train_qa="./data/CaSiNo/train.json"
# new_eval_qa="./data/CaSiNo/valid.json"
# new_min_layer_to_read=5
# new_max_layer_to_read=6

# # Update the configuration file
# sed -i "s/^.*run_name: str = .*/    run_name: str = \"$new_run_name\"/" "$config_file_path"
# sed -i "s/^.*target_model_name: str = .*/    target_model_name: str = \"$new_target_model_name\"/" "$config_file_path"
# sed -i "s/^.*eval_every_n_steps: int = .*/    eval_every_n_steps: int = $new_eval_every_n_steps/" "$config_file_path"
# sed -i "s|^.*eval_qa: str = .*|    eval_qa: str = \"$new_eval_qa\"|" "$config_file_path"
# sed -i "s/^.*min_layer_to_read: int = .*/    min_layer_to_read: int = $new_min_layer_to_read/" "$config_file_path"
# sed -i "s/^.*max_layer_to_read: int = .*/    max_layer_to_read: int = $new_max_layer_to_read/" "$config_file_path"

# # Running the actual experiment
# /srv/scratch/CRUISE/z5517269/miniconda/envs/latentqa/bin/python train.py \
# --target_model_name $new_target_model_name \
# --train_qa $new_train_qa \
# --gradient_accumulation_steps 8 \ 
# --use_wandb \
# --eval_ppl \
# --eval_qa $new_eval_qa \
# --eval_every_n_steps $new_eval_every_n_steps

# # Parameters to change
# new_run_name="CaSiNo Deep Ministral 8B"
# new_target_model_name="mistralai/Ministral-8B-Instruct-2410"
# new_eval_every_n_steps=300
# new_train_qa="./data/CaSiNo/train.json"
# new_eval_qa="./data/CaSiNo/valid.json"
# new_min_layer_to_read=25
# new_max_layer_to_read=26

# # Update the configuration file
# sed -i "s/^.*run_name: str = .*/    run_name: str = \"$new_run_name\"/" "$config_file_path"
# sed -i "s/^.*target_model_name: str = .*/    target_model_name: str = \"$new_target_model_name\"/" "$config_file_path"
# sed -i "s/^.*eval_every_n_steps: int = .*/    eval_every_n_steps: int = $new_eval_every_n_steps/" "$config_file_path"
# sed -i "s|^.*eval_qa: str = .*|    eval_qa: str = \"$new_eval_qa\"|" "$config_file_path"
# sed -i "s/^.*min_layer_to_read: int = .*/    min_layer_to_read: int = $new_min_layer_to_read/" "$config_file_path"
# sed -i "s/^.*max_layer_to_read: int = .*/    max_layer_to_read: int = $new_max_layer_to_read/" "$config_file_path"

# # Running the actual experiment
# /srv/scratch/CRUISE/z5517269/miniconda/envs/latentqa/bin/python train.py \
# --target_model_name $new_target_model_name \
# --train_qa $new_train_qa \
# --gradient_accumulation_steps 8 \ 
# --use_wandb \
# --eval_ppl \
# --eval_qa $new_eval_qa \
# --eval_every_n_steps $new_eval_every_n_steps