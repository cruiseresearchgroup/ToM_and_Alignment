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


# ################################################################################
# ################################### Llama3 1B ##################################
# ################################################################################

# target_model_name="meta-llama/Llama-3.2-1B-Instruct"

# new_run_name="CRAIGSLISTBARGAIN Middle Llama-3 1B"
# new_eval_qa="./data/CRAIGSLISTBARGAIN/test.json"
# new_min_layer_to_read=8
# new_max_layer_to_read=9

# # Update the configuration file
# sed -i "s/^    save_name: str = \".*\"/    save_name: str = \"$new_run_name\"/" "$config_file_path"
# sed -i "s|^    eval_qa: str = \".*\"|    eval_qa: str = \"$new_eval_qa\"|" "$config_file_path"
# sed -i "s/^    min_layer_to_read: int = [0-9]*/    min_layer_to_read: int = $new_min_layer_to_read/" "$config_file_path"
# sed -i "s/^    max_layer_to_read: int = [0-9]*/    max_layer_to_read: int = $new_max_layer_to_read/" "$config_file_path"

# /srv/scratch/CRUISE/z5517269/miniconda/envs/latentqa/bin/python reading.py \
#     --target_model_name $target_model_name \
#     --decoder_model_name /srv/scratch/CRUISE/Mehdi/out/runs/009/checkpoints/epoch4-steps20175-2025-01-14_22-42-48 # path to the best model 

# # Parameters to change
# new_run_name="CRAIGSLISTBARGAIN Shallow Llama-3 1B"
# new_eval_qa="./data/CRAIGSLISTBARGAIN/test.json"
# new_min_layer_to_read=3
# new_max_layer_to_read=4

# # Update the configuration file
# sed -i "s/^    save_name: str = \".*\"/    save_name: str = \"$new_run_name\"/" "$config_file_path"
# sed -i "s|^    eval_qa: str = \".*\"|    eval_qa: str = \"$new_eval_qa\"|" "$config_file_path"
# sed -i "s/^    min_layer_to_read: int = [0-9]*/    min_layer_to_read: int = $new_min_layer_to_read/" "$config_file_path"
# sed -i "s/^    max_layer_to_read: int = [0-9]*/    max_layer_to_read: int = $new_max_layer_to_read/" "$config_file_path"

# /srv/scratch/CRUISE/z5517269/miniconda/envs/latentqa/bin/python reading.py \
#     --target_model_name $target_model_name \
#     --decoder_model_name /srv/scratch/CRUISE/Mehdi/out/runs/010/checkpoints/epoch4-steps20175-2025-01-14_22-59-28 # path to the best model 

# # Parameters to change
# new_run_name="CRAIGSLISTBARGAIN Deep Llama-3 1B"
# new_eval_qa="./data/CRAIGSLISTBARGAIN/test.json"
# new_min_layer_to_read=14
# new_max_layer_to_read=15

# # Update the configuration file
# sed -i "s/^    save_name: str = \".*\"/    save_name: str = \"$new_run_name\"/" "$config_file_path"
# sed -i "s|^    eval_qa: str = \".*\"|    eval_qa: str = \"$new_eval_qa\"|" "$config_file_path"
# sed -i "s/^    min_layer_to_read: int = [0-9]*/    min_layer_to_read: int = $new_min_layer_to_read/" "$config_file_path"
# sed -i "s/^    max_layer_to_read: int = [0-9]*/    max_layer_to_read: int = $new_max_layer_to_read/" "$config_file_path"

# /srv/scratch/CRUISE/z5517269/miniconda/envs/latentqa/bin/python reading.py \
#     --target_model_name $target_model_name \
#     --decoder_model_name /srv/scratch/CRUISE/Mehdi/out/runs/011/checkpoints/epoch4-steps20175-2025-01-14_23-16-07 # path to the best model 

################################################################################
################################### Llama3 3B ##################################
################################################################################
target_model_name="meta-llama/Llama-3.2-3B-Instruct"


# # Parameters to change
# new_run_name="CRAIGSLISTBARGAIN Middle Llama-3 3B"
# new_eval_qa="./data/CRAIGSLISTBARGAIN/test.json"
# new_min_layer_to_read=15
# new_max_layer_to_read=16

# # Update the configuration file
# sed -i "s/^    save_name: str = \".*\"/    save_name: str = \"$new_run_name\"/" "$config_file_path"
# sed -i "s|^    eval_qa: str = \".*\"|    eval_qa: str = \"$new_eval_qa\"|" "$config_file_path"
# sed -i "s/^    min_layer_to_read: int = [0-9]*/    min_layer_to_read: int = $new_min_layer_to_read/" "$config_file_path"
# sed -i "s/^    max_layer_to_read: int = [0-9]*/    max_layer_to_read: int = $new_max_layer_to_read/" "$config_file_path"

# /srv/scratch/CRUISE/z5517269/miniconda/envs/latentqa/bin/python reading.py \
#     --target_model_name $target_model_name \
#     --decoder_model_name /srv/scratch/CRUISE/Mehdi/out/runs/012/checkpoints/epoch4-steps20175-2025-01-14_23-43-50 # path to the best model 

# Parameters to change
new_run_name="CRAIGSLISTBARGAIN Shallow Llama-3 3B"
new_eval_qa="./data/CRAIGSLISTBARGAIN/test.json"
new_min_layer_to_read=5
new_max_layer_to_read=6

# Update the configuration file
sed -i "s/^    save_name: str = \".*\"/    save_name: str = \"$new_run_name\"/" "$config_file_path"
sed -i "s|^    eval_qa: str = \".*\"|    eval_qa: str = \"$new_eval_qa\"|" "$config_file_path"
sed -i "s/^    min_layer_to_read: int = [0-9]*/    min_layer_to_read: int = $new_min_layer_to_read/" "$config_file_path"
sed -i "s/^    max_layer_to_read: int = [0-9]*/    max_layer_to_read: int = $new_max_layer_to_read/" "$config_file_path"

/srv/scratch/CRUISE/z5517269/miniconda/envs/latentqa/bin/python reading.py \
    --target_model_name $target_model_name \
    --decoder_model_name /srv/scratch/CRUISE/Mehdi/out/runs/013/checkpoints/epoch4-steps20175-2025-01-15_00-11-59 # path to the best model 

# # Parameters to change
# new_run_name="CRAIGSLISTBARGAIN Deep Llama-3 3B"
# new_eval_qa="./data/CRAIGSLISTBARGAIN/test.json"
# new_min_layer_to_read=25
# new_max_layer_to_read=26

# # Update the configuration file
# sed -i "s/^    save_name: str = \".*\"/    save_name: str = \"$new_run_name\"/" "$config_file_path"
# sed -i "s|^    eval_qa: str = \".*\"|    eval_qa: str = \"$new_eval_qa\"|" "$config_file_path"
# sed -i "s/^    min_layer_to_read: int = [0-9]*/    min_layer_to_read: int = $new_min_layer_to_read/" "$config_file_path"
# sed -i "s/^    max_layer_to_read: int = [0-9]*/    max_layer_to_read: int = $new_max_layer_to_read/" "$config_file_path"

# /srv/scratch/CRUISE/z5517269/miniconda/envs/latentqa/bin/python reading.py \
#     --target_model_name $target_model_name \
#     --decoder_model_name  /srv/scratch/CRUISE/Mehdi/out/runs/014/checkpoints/epoch4-steps20175-2025-01-15_00-39-59 # path to the best model 


# ################################################################################
# ################################### Llama3 8B ##################################
# ################################################################################
# target_model_name="meta-llama/Meta-Llama-3-8B-Instruct"


# # Parameters to change
# new_run_name="CRAIGSLISTBARGAIN Middle Llama-3 8B"
# new_eval_qa="./data/CRAIGSLISTBARGAIN/test.json"
# new_min_layer_to_read=15
# new_max_layer_to_read=16

# # Update the configuration file
# sed -i "s/^    save_name: str = \".*\"/    save_name: str = \"$new_run_name\"/" "$config_file_path"
# sed -i "s|^    eval_qa: str = \".*\"|    eval_qa: str = \"$new_eval_qa\"|" "$config_file_path"
# sed -i "s/^    min_layer_to_read: int = [0-9]*/    min_layer_to_read: int = $new_min_layer_to_read/" "$config_file_path"
# sed -i "s/^    max_layer_to_read: int = [0-9]*/    max_layer_to_read: int = $new_max_layer_to_read/" "$config_file_path"

# /srv/scratch/CRUISE/z5517269/miniconda/envs/latentqa/bin/python reading.py \
#     --target_model_name $target_model_name \
#     --decoder_model_name  /srv/scratch/CRUISE/Mehdi/out/runs/015/checkpoints/epoch2-steps12105-2025-01-15_00-59-39 # path to the best model 

# # Parameters to change
# new_run_name="CRAIGSLISTBARGAIN Shallow Llama-3 8B"
# new_eval_qa="./data/CRAIGSLISTBARGAIN/test.json"
# new_min_layer_to_read=5
# new_max_layer_to_read=6

# # Update the configuration file
# sed -i "s/^    save_name: str = \".*\"/    save_name: str = \"$new_run_name\"/" "$config_file_path"
# sed -i "s|^    eval_qa: str = \".*\"|    eval_qa: str = \"$new_eval_qa\"|" "$config_file_path"
# sed -i "s/^    min_layer_to_read: int = [0-9]*/    min_layer_to_read: int = $new_min_layer_to_read/" "$config_file_path"
# sed -i "s/^    max_layer_to_read: int = [0-9]*/    max_layer_to_read: int = $new_max_layer_to_read/" "$config_file_path"

# /srv/scratch/CRUISE/z5517269/miniconda/envs/latentqa/bin/python reading.py \
#     --target_model_name $target_model_name \
#     --decoder_model_name  /srv/scratch/CRUISE/Mehdi/out/runs/016/checkpoints/epoch2-steps12105-2025-01-15_01-32-17 # path to the best model 

# # Parameters to change
# new_run_name="CRAIGSLISTBARGAIN Deep Llama-3 8B"
# new_eval_qa="./data/CRAIGSLISTBARGAIN/test.json"
# new_min_layer_to_read=25
# new_max_layer_to_read=26

# # Update the configuration file
# sed -i "s/^    save_name: str = \".*\"/    save_name: str = \"$new_run_name\"/" "$config_file_path"
# sed -i "s|^    eval_qa: str = \".*\"|    eval_qa: str = \"$new_eval_qa\"|" "$config_file_path"
# sed -i "s/^    min_layer_to_read: int = [0-9]*/    min_layer_to_read: int = $new_min_layer_to_read/" "$config_file_path"
# sed -i "s/^    max_layer_to_read: int = [0-9]*/    max_layer_to_read: int = $new_max_layer_to_read/" "$config_file_path"

# /srv/scratch/CRUISE/z5517269/miniconda/envs/latentqa/bin/python reading.py \
#     --target_model_name $target_model_name \
#     --decoder_model_name  /srv/scratch/CRUISE/Mehdi/out/runs/017/checkpoints/epoch2-steps12105-2025-01-15_02-04-37 # path to the best model 

##############################################################################
################################# Ministral 8B ###############################
##############################################################################
# target_model_name="mistralai/Ministral-8B-Instruct-2410"


# # Parameters to change
# new_run_name="CRAIGSLISTBARGAIN Middle Ministral 8B"
# new_eval_qa="./data/CRAIGSLISTBARGAIN/test.json"
# new_min_layer_to_read=15
# new_max_layer_to_read=16

# # Update the configuration file
# sed -i "s/^    save_name: str = \".*\"/    save_name: str = \"$new_run_name\"/" "$config_file_path"
# sed -i "s|^    eval_qa: str = \".*\"|    eval_qa: str = \"$new_eval_qa\"|" "$config_file_path"
# sed -i "s/^    min_layer_to_read: int = [0-9]*/    min_layer_to_read: int = $new_min_layer_to_read/" "$config_file_path"
# sed -i "s/^    max_layer_to_read: int = [0-9]*/    max_layer_to_read: int = $new_max_layer_to_read/" "$config_file_path"

# /srv/scratch/CRUISE/z5517269/miniconda/envs/latentqa/bin/python reading.py \
#     --target_model_name $target_model_name \
#     --decoder_model_name  /srv/scratch/CRUISE/Mehdi/out/runs/036/checkpoints/epoch1-steps1800-2025-01-14_14-31-00 # path to the best model 

# # Parameters to change
# new_run_name="CRAIGSLISTBARGAIN Shallow Ministral 8B"
# new_eval_qa="./data/CRAIGSLISTBARGAIN/test.json"
# new_min_layer_to_read=5
# new_max_layer_to_read=6

# # Update the configuration file
# sed -i "s/^    save_name: str = \".*\"/    save_name: str = \"$new_run_name\"/" "$config_file_path"
# sed -i "s|^    eval_qa: str = \".*\"|    eval_qa: str = \"$new_eval_qa\"|" "$config_file_path"
# sed -i "s/^    min_layer_to_read: int = [0-9]*/    min_layer_to_read: int = $new_min_layer_to_read/" "$config_file_path"
# sed -i "s/^    max_layer_to_read: int = [0-9]*/    max_layer_to_read: int = $new_max_layer_to_read/" "$config_file_path"

# /srv/scratch/CRUISE/z5517269/miniconda/envs/latentqa/bin/python reading.py \
#     --target_model_name $target_model_name \
#     --decoder_model_name  /srv/scratch/CRUISE/Mehdi/out/runs/037/checkpoints/epoch1-steps1800-2025-01-14_14-40-22 # path to the best model 

# # Parameters to change
# new_run_name="CRAIGSLISTBARGAIN Deep Ministral 8B"
# new_eval_qa="./data/CRAIGSLISTBARGAIN/test.json"
# new_min_layer_to_read=25
# new_max_layer_to_read=26

# # Update the configuration file
# sed -i "s/^    save_name: str = \".*\"/    save_name: str = \"$new_run_name\"/" "$config_file_path"
# sed -i "s|^    eval_qa: str = \".*\"|    eval_qa: str = \"$new_eval_qa\"|" "$config_file_path"
# sed -i "s/^    min_layer_to_read: int = [0-9]*/    min_layer_to_read: int = $new_min_layer_to_read/" "$config_file_path"
# sed -i "s/^    max_layer_to_read: int = [0-9]*/    max_layer_to_read: int = $new_max_layer_to_read/" "$config_file_path"

# /srv/scratch/CRUISE/z5517269/miniconda/envs/latentqa/bin/python reading.py \
#     --target_model_name $target_model_name \
#     --decoder_model_name  /srv/scratch/CRUISE/Mehdi/out/runs/038/checkpoints/epoch3-steps3600-2025-01-14_14-53-12 # path to the best model 
