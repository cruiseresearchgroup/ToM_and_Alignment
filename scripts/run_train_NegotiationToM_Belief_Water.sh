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
################################### Llama3 8B ##################################
################################################################################

# Parameters to change
new_run_name="NegotiationToM Belief 8B"
new_target_model_name="meta-llama/Meta-Llama-3-8B-Instruct"
new_eval_every_n_steps=334
new_train_qa="./data/NegotiationToM/train.json"
new_eval_qa="./data/NegotiationToM/valid.json"
new_steer_component="Belief"
new_min_layer_to_read=15
new_max_layer_to_read=16

# Update the configuration file
sed -i "s/^.*run_name: str = .*/    run_name: str = \"$new_run_name\"/" "$config_file_path"
sed -i "s/^.*target_model_name: str = .*/    target_model_name: str = \"$new_target_model_name\"/" "$config_file_path"
sed -i "s/^.*eval_every_n_steps: int = .*/    eval_every_n_steps: int = $new_eval_every_n_steps/" "$config_file_path"
sed -i "s|^.*eval_qa: str = .*|    eval_qa: str = \"$new_eval_qa\"|" "$config_file_path"
sed -i "s/^.*min_layer_to_read: int = .*/    min_layer_to_read: int = $new_min_layer_to_read/" "$config_file_path"
sed -i "s/^.*max_layer_to_read: int = .*/    max_layer_to_read: int = $new_max_layer_to_read/" "$config_file_path"
sed -i "s/^.*steer_component: int = .*/    steer_component: str = $new_steer_component/" "$config_file_path"

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
new_run_name="NegotiationToM Desire 8B"
new_target_model_name="meta-llama/Meta-Llama-3-8B-Instruct"
new_eval_every_n_steps=334
new_train_qa="./data/NegotiationToM/train.json"
new_eval_qa="./data/NegotiationToM/valid.json"
new_steer_component="Desire"
new_min_layer_to_read=15
new_max_layer_to_read=16

# Update the configuration file
sed -i "s/^.*run_name: str = .*/    run_name: str = \"$new_run_name\"/" "$config_file_path"
sed -i "s/^.*target_model_name: str = .*/    target_model_name: str = \"$new_target_model_name\"/" "$config_file_path"
sed -i "s/^.*eval_every_n_steps: int = .*/    eval_every_n_steps: int = $new_eval_every_n_steps/" "$config_file_path"
sed -i "s|^.*eval_qa: str = .*|    eval_qa: str = \"$new_eval_qa\"|" "$config_file_path"
sed -i "s/^.*min_layer_to_read: int = .*/    min_layer_to_read: int = $new_min_layer_to_read/" "$config_file_path"
sed -i "s/^.*max_layer_to_read: int = .*/    max_layer_to_read: int = $new_max_layer_to_read/" "$config_file_path"
sed -i "s/^.*steer_component: int = .*/    steer_component: str = $new_steer_component/" "$config_file_path"

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
new_run_name="NegotiationToM Intention 8B"
new_target_model_name="meta-llama/Meta-Llama-3-8B-Instruct"
new_eval_every_n_steps=334
new_train_qa="./data/NegotiationToM/train.json"
new_eval_qa="./data/NegotiationToM/valid.json"
new_steer_component="Intention"
new_min_layer_to_read=15
new_max_layer_to_read=16

# Update the configuration file
sed -i "s/^.*run_name: str = .*/    run_name: str = \"$new_run_name\"/" "$config_file_path"
sed -i "s/^.*target_model_name: str = .*/    target_model_name: str = \"$new_target_model_name\"/" "$config_file_path"
sed -i "s/^.*eval_every_n_steps: int = .*/    eval_every_n_steps: int = $new_eval_every_n_steps/" "$config_file_path"
sed -i "s|^.*eval_qa: str = .*|    eval_qa: str = \"$new_eval_qa\"|" "$config_file_path"
sed -i "s/^.*min_layer_to_read: int = .*/    min_layer_to_read: int = $new_min_layer_to_read/" "$config_file_path"
sed -i "s/^.*max_layer_to_read: int = .*/    max_layer_to_read: int = $new_max_layer_to_read/" "$config_file_path"
sed -i "s/^.*steer_component: int = .*/    steer_component: str = $new_steer_component/" "$config_file_path"

# Running the actual experiment
/srv/scratch/CRUISE/z5517269/miniconda/envs/latentqa/bin/python train.py \
--target_model_name $new_target_model_name \
--train_qa $new_train_qa \
--gradient_accumulation_steps 8 \ 
--use_wandb \
--eval_ppl \
--eval_qa $new_eval_qa \
--eval_every_n_steps $new_eval_every_n_steps
