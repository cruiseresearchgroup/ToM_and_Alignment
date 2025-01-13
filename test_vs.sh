#!/bin/bash -l

# PLEASE MAKE SURE YOU CHANGE THE FOLLOWING QSUB SETTINGS TO SUIT YOUR NEEDS
#
#$ -j y
#$ -e $JOB_ID_$JOB_NAME.out
#$ -o $JOB_ID_$JOB_NAME.out
#$ -cwd
#$ -N vscode-tunnel
#$ -l mem=15G,jobfs=10G
#$ -V
#

module load vscode/vscode-cli

launch_tunnel
