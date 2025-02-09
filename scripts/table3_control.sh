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

# Configuration file path
config_file_path="./lit/configs/steer_config.py"

export RANK=0
export WORLD_SIZE=1
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=12348
export RAYON_NUM_THREADS=48
export HF_HOME=/srv/scratch/CRUISE/Mehdi/HF

################################################################################
################################### Llama3 8B ##################################
################################################################################

target_model_name="meta-llama/Meta-Llama-3-8B-Instruct"


# # Parameters to change
# new_steer_component="Belief"
# new_steer_label="Water"

# # Update the configuration file
# sed -i "s/^    target_model_name: str = \".*\"/    target_model_name: str = \"$target_model_name\"/" "$config_file_path"
# sed -i "s/^    steer_component: str = \".*\"/    steer_component: str = \"$new_steer_component\"/" "$config_file_path"
# sed -i "s|^    steer_label: str = \".*\"|    steer_label: str = \"$new_steer_label\"|" "$config_file_path"



# /srv/scratch/CRUISE/z5517269/miniconda/envs/latentqa/bin/python control.py \
#     --decoder_model_name /srv/scratch/CRUISE/Mehdi/out/runs/026/checkpoints/epoch4-steps6675-2025-01-20_15-31-02

# # Parameters to change
# new_steer_component="Belief"
# new_steer_label="Food"

# # Update the configuration file
# sed -i "s/^    target_model_name: str = \".*\"/    target_model_name: str = \"$target_model_name\"/" "$config_file_path"
# sed -i "s/^    steer_component: str = \".*\"/    steer_component: str = \"$new_steer_component\"/" "$config_file_path"
# sed -i "s|^    steer_label: str = \".*\"|    steer_label: str = \"$new_steer_label\"|" "$config_file_path"



# /srv/scratch/CRUISE/z5517269/miniconda/envs/latentqa/bin/python control.py \
#     --decoder_model_name /srv/scratch/CRUISE/Mehdi/out/runs/026/checkpoints/epoch4-steps6675-2025-01-20_15-31-02


# # Parameters to change
# new_steer_component="Belief"
# new_steer_label="Firewood"

# # Update the configuration file
# sed -i "s/^    target_model_name: str = \".*\"/    target_model_name: str = \"$target_model_name\"/" "$config_file_path"
# sed -i "s/^    steer_component: str = \".*\"/    steer_component: str = \"$new_steer_component\"/" "$config_file_path"
# sed -i "s|^    steer_label: str = \".*\"|    steer_label: str = \"$new_steer_label\"|" "$config_file_path"



# /srv/scratch/CRUISE/z5517269/miniconda/envs/latentqa/bin/python control.py \
#     --decoder_model_name /srv/scratch/CRUISE/Mehdi/out/runs/026/checkpoints/epoch4-steps6675-2025-01-20_15-31-02

# # Parameters to change
# new_steer_component="Desire"
# new_steer_label="Water"

# # Update the configuration file
# sed -i "s/^    target_model_name: str = \".*\"/    target_model_name: str = \"$target_model_name\"/" "$config_file_path"
# sed -i "s/^    steer_component: str = \".*\"/    steer_component: str = \"$new_steer_component\"/" "$config_file_path"
# sed -i "s|^    steer_label: str = \".*\"|    steer_label: str = \"$new_steer_label\"|" "$config_file_path"



# /srv/scratch/CRUISE/z5517269/miniconda/envs/latentqa/bin/python control.py \
#     --decoder_model_name /srv/scratch/CRUISE/Mehdi/out/runs/024/checkpoints/epoch4-steps6675-2025-01-20_15-47-58

# # Parameters to change
# new_steer_component="Desire"
# new_steer_label="Food"

# # Update the configuration file
# sed -i "s/^    target_model_name: str = \".*\"/    target_model_name: str = \"$target_model_name\"/" "$config_file_path"
# sed -i "s/^    steer_component: str = \".*\"/    steer_component: str = \"$new_steer_component\"/" "$config_file_path"
# sed -i "s|^    steer_label: str = \".*\"|    steer_label: str = \"$new_steer_label\"|" "$config_file_path"



# /srv/scratch/CRUISE/z5517269/miniconda/envs/latentqa/bin/python control.py \
#     --decoder_model_name /srv/scratch/CRUISE/Mehdi/out/runs/024/checkpoints/epoch4-steps6675-2025-01-20_15-47-58

# # Parameters to change
# new_steer_component="Desire"
# new_steer_label="Firewood"

# # Update the configuration file
# sed -i "s/^    target_model_name: str = \".*\"/    target_model_name: str = \"$target_model_name\"/" "$config_file_path"
# sed -i "s/^    steer_component: str = \".*\"/    steer_component: str = \"$new_steer_component\"/" "$config_file_path"
# sed -i "s|^    steer_label: str = \".*\"|    steer_label: str = \"$new_steer_label\"|" "$config_file_path"



# /srv/scratch/CRUISE/z5517269/miniconda/envs/latentqa/bin/python control.py \
#     --decoder_model_name /srv/scratch/CRUISE/Mehdi/out/runs/024/checkpoints/epoch4-steps6675-2025-01-20_15-47-58


# # Parameters to change
# new_steer_component="Intention"
# new_steer_label="Show-Empathy"

# # Update the configuration file
# sed -i "s/^    target_model_name: str = \".*\"/    target_model_name: str = \"$target_model_name\"/" "$config_file_path"
# sed -i "s/^    steer_component: str = \".*\"/    steer_component: str = \"$new_steer_component\"/" "$config_file_path"
# sed -i "s|^    steer_label: str = \".*\"|    steer_label: str = \"$new_steer_label\"|" "$config_file_path"



# /srv/scratch/CRUISE/z5517269/miniconda/envs/latentqa/bin/python control.py \
#     --decoder_model_name /srv/scratch/CRUISE/Mehdi/out/runs/025/checkpoints/epoch4-steps6675-2025-01-20_16-04-30

# # Parameters to change
# new_steer_component="Intention"
# new_steer_label="Promote-Coordination"

# # Update the configuration file
# sed -i "s/^    target_model_name: str = \".*\"/    target_model_name: str = \"$target_model_name\"/" "$config_file_path"
# sed -i "s/^    steer_component: str = \".*\"/    steer_component: str = \"$new_steer_component\"/" "$config_file_path"
# sed -i "s|^    steer_label: str = \".*\"|    steer_label: str = \"$new_steer_label\"|" "$config_file_path"



# /srv/scratch/CRUISE/z5517269/miniconda/envs/latentqa/bin/python control.py \
#     --decoder_model_name /srv/scratch/CRUISE/Mehdi/out/runs/025/checkpoints/epoch4-steps6675-2025-01-20_16-04-30

# # Parameters to change
# new_steer_component="Intention"
# new_steer_label="Build-Rapport"

# # Update the configuration file
# sed -i "s/^    target_model_name: str = \".*\"/    target_model_name: str = \"$target_model_name\"/" "$config_file_path"
# sed -i "s/^    steer_component: str = \".*\"/    steer_component: str = \"$new_steer_component\"/" "$config_file_path"
# sed -i "s|^    steer_label: str = \".*\"|    steer_label: str = \"$new_steer_label\"|" "$config_file_path"



# /srv/scratch/CRUISE/z5517269/miniconda/envs/latentqa/bin/python control.py \
#     --decoder_model_name /srv/scratch/CRUISE/Mehdi/out/runs/025/checkpoints/epoch4-steps6675-2025-01-20_16-04-30

# # Parameters to change
# new_steer_component="Intention"
# new_steer_label="Discover-Preference"

# # Update the configuration file
# sed -i "s/^    target_model_name: str = \".*\"/    target_model_name: str = \"$target_model_name\"/" "$config_file_path"
# sed -i "s/^    steer_component: str = \".*\"/    steer_component: str = \"$new_steer_component\"/" "$config_file_path"
# sed -i "s|^    steer_label: str = \".*\"|    steer_label: str = \"$new_steer_label\"|" "$config_file_path"



# /srv/scratch/CRUISE/z5517269/miniconda/envs/latentqa/bin/python control.py \
#     --decoder_model_name /srv/scratch/CRUISE/Mehdi/out/runs/025/checkpoints/epoch4-steps6675-2025-01-20_16-04-30

# # Parameters to change
# new_steer_component="Intention"
# new_steer_label="Describe-Need"

# # Update the configuration file
# sed -i "s/^    target_model_name: str = \".*\"/    target_model_name: str = \"$target_model_name\"/" "$config_file_path"
# sed -i "s/^    steer_component: str = \".*\"/    steer_component: str = \"$new_steer_component\"/" "$config_file_path"
# sed -i "s|^    steer_label: str = \".*\"|    steer_label: str = \"$new_steer_label\"|" "$config_file_path"



# /srv/scratch/CRUISE/z5517269/miniconda/envs/latentqa/bin/python control.py \
#     --decoder_model_name /srv/scratch/CRUISE/Mehdi/out/runs/025/checkpoints/epoch4-steps6675-2025-01-20_16-04-30

    
# # Parameters to change
# new_steer_component="Intention"
# new_steer_label="Show-Empathy"

# # Update the configuration file
# sed -i "s/^    target_model_name: str = \".*\"/    target_model_name: str = \"$target_model_name\"/" "$config_file_path"
# sed -i "s/^    steer_component: str = \".*\"/    steer_component: str = \"$new_steer_component\"/" "$config_file_path"
# sed -i "s|^    steer_label: str = \".*\"|    steer_label: str = \"$new_steer_label\"|" "$config_file_path"



# /srv/scratch/CRUISE/z5517269/miniconda/envs/latentqa/bin/python control.py \
#     --decoder_model_name /srv/scratch/CRUISE/Mehdi/out/runs/025/checkpoints/epoch4-steps6675-2025-01-20_16-04-30