#!/bin/bash
#SBATCH --job-name=SSD_recycle
#SBATCH --output=SSD_recycle.log
#SBATCH --nodes=1
#SBATCH -n 12
#SBATCH --mem=16gb
#SBATCH --gres=gpu:2
#SBATCH -t 12:00:00

date;hostname;pwd

################################
# How to use this script?
# 1) On the cluster head node, batch submit this script: sbatch tensorboard_begin.sh
# 2) Open the tensorboard.log file and copy the line "SSH tunnel command", starting from "ssh - NL ..."
# 3) On your local computer (laptop) terminal, paste the above command to build the ssh tunnel.
# 4) On your local computer, open a browser and visit: http://localhost:6008
################################

export XDG_RUNTIME_DIR=""

#port=$(shuf -i 6000-7000 -n 1)
#echo -e "\nStarting Jupyter on port ${port} on the $(hostname) server."
#echo -e "\nSSH tunnel command: ssh -NL 6600:$(hostname):${port} ${USER}@ace.wpi.edu"
#echo -e "\nLocal URI: http://localhost:6600"

# Change this line to match your virtualenv
source /home/cmburgul/singularity/dr/envpy36/bin/activate

#jupyter-notebook --notebook-dir=/home/abiyer --no-browser --port=${port} --ip='0.0.0.0'

python multistart_multigoal_train.py
