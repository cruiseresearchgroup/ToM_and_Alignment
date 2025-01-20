#
#$ -j y
#$ -e $JOB_ID_$JOB_NAME.out
#$ -o $JOB_ID_$JOB_NAME.out
#$ -cwd
#$ -N latentqa
#$ -pe smp 64
#$ -l mem=100G,jobfs=180G,ngpus=2,gpu_model=H100_NVL
#$ -V
#$ -P CRUISE

# Configuration file path
config_file_path="./lit/configs/interpret_config.py"

export RANK=0
export WORLD_SIZE=1
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=12348
export RAYON_NUM_THREADS=48
export HF_HOME=/srv/scratch/CRUISE/Mehdi/HF


################################################################################
################################### Llama3 3B ##################################
################################################################################
target_model_name="meta-llama/Llama-3.2-3B-Instruct"


# Parameters to change
new_run_name="FANTOM Middle Llama-3 3B"
new_eval_qa="./data/FANTOM/test.json"
new_min_layer_to_read=15
new_max_layer_to_read=16

# Update the configuration file
sed -i "s/^    save_name: str = \".*\"/    save_name: str = \"$new_run_name\"/" "$config_file_path"
sed -i "s|^    eval_qa: str = \".*\"|    eval_qa: str = \"$new_eval_qa\"|" "$config_file_path"
sed -i "s/^    min_layer_to_read: int = [0-9]*/    min_layer_to_read: int = $new_min_layer_to_read/" "$config_file_path"
sed -i "s/^    max_layer_to_read: int = [0-9]*/    max_layer_to_read: int = $new_max_layer_to_read/" "$config_file_path"

/srv/scratch/CRUISE/z5517269/miniconda/envs/latentqa/bin/python reading.py \
    --target_model_name $target_model_name \
    --decoder_model_name /srv/scratch/CRUISE/Mehdi/out/runs/019/checkpoints/epoch3-steps28892-2025-01-15_16-49-12 # path to the best model 

# # Parameters to change
# new_run_name="NegotiationToM Middle Llama-3 3B"
# new_eval_qa="./data/NegotiationToM/test.json"
# new_min_layer_to_read=15
# new_max_layer_to_read=16

# # Update the configuration file
# sed -i "s/^    save_name: str = \".*\"/    save_name: str = \"$new_run_name\"/" "$config_file_path"
# sed -i "s|^    eval_qa: str = \".*\"|    eval_qa: str = \"$new_eval_qa\"|" "$config_file_path"
# sed -i "s/^    min_layer_to_read: int = [0-9]*/    min_layer_to_read: int = $new_min_layer_to_read/" "$config_file_path"
# sed -i "s/^    max_layer_to_read: int = [0-9]*/    max_layer_to_read: int = $new_max_layer_to_read/" "$config_file_path"

# /srv/scratch/CRUISE/z5517269/miniconda/envs/latentqa/bin/python reading.py \
#     --target_model_name $target_model_name \
#     --decoder_model_name /srv/scratch/CRUISE/Mehdi/out/runs/018/checkpoints/epoch4-steps6675-2025-01-15_15-00-52 # path to the best model 

################################################################################
################################### Llama3 8B ##################################
################################################################################
target_model_name="meta-llama/Meta-Llama-3-8B-Instruct"


# Parameters to change
new_run_name="FANTOM Middle Llama-3 8B"
new_eval_qa="./data/FANTOM/test.json"
new_min_layer_to_read=15
new_max_layer_to_read=16

# Update the configuration file
sed -i "s/^    save_name: str = \".*\"/    save_name: str = \"$new_run_name\"/" "$config_file_path"
sed -i "s|^    eval_qa: str = \".*\"|    eval_qa: str = \"$new_eval_qa\"|" "$config_file_path"
sed -i "s/^    min_layer_to_read: int = [0-9]*/    min_layer_to_read: int = $new_min_layer_to_read/" "$config_file_path"
sed -i "s/^    max_layer_to_read: int = [0-9]*/    max_layer_to_read: int = $new_max_layer_to_read/" "$config_file_path"

/srv/scratch/CRUISE/z5517269/miniconda/envs/latentqa/bin/python reading.py \
    --target_model_name $target_model_name \
    --decoder_model_name  /srv/scratch/CRUISE/Mehdi/out/runs/021/checkpoints/epoch2-steps21669-2025-01-15_20-02-25 # path to the best model 

# # Parameters to change
# new_run_name="NegotiationToM Middle Llama-3 8B"
# new_eval_qa="./data/NegotiationToM/test.json"
# new_min_layer_to_read=15
# new_max_layer_to_read=16

# # Update the configuration file
# sed -i "s/^    save_name: str = \".*\"/    save_name: str = \"$new_run_name\"/" "$config_file_path"
# sed -i "s|^    eval_qa: str = \".*\"|    eval_qa: str = \"$new_eval_qa\"|" "$config_file_path"
# sed -i "s/^    min_layer_to_read: int = [0-9]*/    min_layer_to_read: int = $new_min_layer_to_read/" "$config_file_path"
# sed -i "s/^    max_layer_to_read: int = [0-9]*/    max_layer_to_read: int = $new_max_layer_to_read/" "$config_file_path"

# /srv/scratch/CRUISE/z5517269/miniconda/envs/latentqa/bin/python reading.py \
#     --target_model_name $target_model_name \
#     --decoder_model_name  /srv/scratch/CRUISE/Mehdi/out/runs/020/checkpoints/epoch4-steps6675-2025-01-15_17-33-16 # path to the best model 

