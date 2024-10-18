# Brain Decoding
* This is the repo for our Brain Decoding project
* Aim:
    - Build a **Multi-Subject** Brain Decoding Model

* Upstream Task: Brain decoder pretraining with LARGE DATASET, generate subject-invariant and generalizable latent embedding.
* Downstream Task:
    - fMRI-to-Img Reconstruction/Retrieval
    - Or other brain decoding tasks..


# HPC Simple Usage
Always ask hpc@nyu.edu if you have technique difficulties. 

## Access to HPC
open [account](https://sites.google.com/nyu.edu/nyu-hpc/accessing-hpc/getting-and-renewing-an-account?authuser=0)
how to [access](https://sites.google.com/nyu.edu/nyu-hpc/accessing-hpc?authuser=0)

## how to login smoothly via nickname and without password
in your local pc, do `vim ~/.ssh/config` write something like:

```
Host *
  ForwardAgent yes
  ServerAliveInterval 60
  ServerAliveCountMax 60
Host *.hpc.nyu.edu
  StrictHostKeyChecking no
  UserKnownHostsFile /dev/null
  LogLevel ERROR
Host hpcgwtunnel
  HostName gw.hpc.nyu.edu
  ForwardX11 no
  LocalForward 8027 greene.hpc.nyu.edu:22
  User xc1490
Host greene
  HostName localhost
  Port 8027
  ForwardX11 yes
  StrictHostKeyChecking no
  UserKnownHostsFile /dev/null
  LogLevel ERROR
  User xc1490
```
then login using `ssh hpcgwtunnel` and keep the window open
then you could login using `ssh greene` in another window

To avoid password, do this on your pc:`ssh-keygen`

Then Append the contents of the file `~/.ssh/id_rsa.pub` created on the client to the end of the file `~/.ssh/authorized_keys` on the remote user at the remote server.


## Quota Limit
- if you do not have enough quota, ask hpc to increase.
- Do not store too many things in /home. Avoid installing packages in /home/.local
- Refer to [here](https://sites.google.com/nyu.edu/nyu-hpc/hpc-systems/hpc-storage/best-practices?authuser=0)

```
$ myquota
Hostname: log-1 at Sun Mar 21 21:59:08 EDT 2021
Filesystem   Environment   Backed up?   Allocation       Current Usage
Space        Variable      /Flushed?    Space / Files    Space(%) / Files(%)
/home        $HOME         Yes/No       50.0GB/30.0K       8.96GB(17.91%)/33000(110.00%)
/scratch     $SCRATCH      No/Yes        5.0TB/1.0M     811.09GB(15.84%)/2437(0.24%)
/archive     $ARCHIVE      Yes/No        2.0TB/20.0K       0.00GB(0.00%)/1(0.00%)
/vast        $VAST         No/Yes        2.0TB/5.0M        0.00GB(0.00%)/1(0.00%)
```

## Environment Setup:
- You should always use singularity to run your program! It is something similar to docker which have an ext image file which is writable by yourself and a sif file from hpc.
- 
- Follow the [Guide](https://sites.google.com/nyu.edu/nyu-hpc/hpc-systems/greene/software/singularity-with-miniconda?authuser=0)
- The singularity image for *swapVAE* is located at: /scratch/cl6707/dl-env/ddsp.ext3
- Check more details in the website above

Some examples:

If you want to change something in your image, use `rw`, then you could install packages.

```
singularity exec --overlay /scratch/cl6707/dl-env/ddsp.ext3:rw /scratch/work/public/singularity/cuda11.0-cudnn8-devel-ubuntu18.04.sif /bin/bash
source /ext3/env.sh
```

## Running jupyter notebook using singularity

Example (if you want GPU), if you only need CPU, remove the line `#SBATCH --gres=gpu:1`

```
#!/bin/bash

#SBATCH --job-name=jpfair
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=24GB
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:1
#SBATCH --account=pr_60_tandon_advanced


module purge
#module load tensorrt/6.0.1.5
#module load libsndfile/intel/1.0.28
#module load ffmpeg/intel/3.2.2

port=$(shuf -i 10000-65500 -n 1)

/usr/bin/ssh -N -f -R $port:localhost:$port log-1
/usr/bin/ssh -N -f -R $port:localhost:$port log-2
/usr/bin/ssh -N -f -R $port:localhost:$port log-3

cat<<EOF

Jupyter server is running on: $(hostname)
Job starts at: $(date)

Step 1 :

If you are working in NYU campus, please open an iTerm window, run command

ssh -L $port:localhost:$port $USER@greene.hpc.nyu.edu

If you are working off campus, you should already have ssh tunneling setup through HPC bastion host,
that you can directly login to greene with command

ssh $USER@greene

Please open an iTerm window, run command

ssh -L $port:localhost:$port $USER@greene

Step 2:

Keep the iTerm windows in the previouse step open. Now open browser, find the line with

The Jupyter Notebook is running at: $(hostname)

the URL is something: http://localhost:${port}/?token=XXXXXXXX (see your token below)

you should be able to connect to jupyter notebook running remotly on greene compute node with above url

EOF

unset XDG_RUNTIME_DIR
if [ "$SLURM_JOBTMP" != "" ]; then
    export XDG_RUNTIME_DIR=$SLURM_JOBTMP
fi

overlay_ext3=/scratch/xc1490/apps/overlay-50G-10M.ext3
singularity exec --nv \
    --overlay ${overlay_ext3}:ro \
    /scratch/work/public/singularity/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif \
    /bin/bash -c "
source /ext3/env.sh
jupyter notebook --no-browser --port $port --notebook-dir=$(pwd)
"
```


## OOD and VS-CODE
- [ ] TODO

## setting alias

`vi ~/.bash_profile` or zshrc if you use zsh. You could set some alias for convenience. For example `alias jpecog='sbatch /scratch/xc1490/apps/run-jupyter_ecog.sbatch'`




## open your folder access to someone else

```
setfacl -m u:netid:rx /scratch/xc1490/ECoG_Shared_Data/LD_data_extracted/meta_data
```

## touch unused files

`/scratch` folder will not be backed up. if hpc sends you email something like this:
>  There are 19793 files in this folder that have not been accessed for more than 60 days. According to HPC policy, these files will be purged after one week from today. The list of files is available on Greene cluster, /scratch/cleanup/60days-files/20230701/xc1490

Then you could touch the file to avoid deleting them by:

```
while read p; do
  touch  $p
done </scratch/cleanup/60days-files/20230701/xc1490
```


## submit job example

```
#!/bin/bash

#SBATCH --job-name=swap869
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=36GB
#SBATCH --time=24:59:00
#SBATCH --gres=gpu:1
#SBATCH --account=pr_60_tandon_advanced
overlay_ext3=/home/xc1490/home/apps/ddsp.ext3
singularity exec --nv \
  --overlay ${overlay_ext3}:ro \
  /scratch/work/public/singularity/cuda11.0-cudnn8-devel-ubuntu18.04.sif \
  /bin/bash -c "
source /ext3/env.sh
cd /scratch/xc1490/projects/ecog/SSL/Medical_SSL/bin_swap_pretrain
python train.py --params
  ```
## Slurm commands:

Refer to [here](https://sites.google.com/nyu.edu/nyu-hpc/training-support/general-hpc-topics/slurm-submitting-jobs)
```
# Interactive mode (GPU)
srun -t 2:00:00 --mem=16000 --gres=gpu:1 --pty /bin/bash # no specific GPU type
srun -t 2:00:00 --mem=16000 --gres=gpu:v100:1 --pty /bin/bash
srun -t 2:00:00 --mem=16000 --gres=gpu:a100:1 --pty /bin/bash
srun -t 2:00:00 --mem=16000 --gres=gpu:rtx8000:1 --pty /bin/bash

# Submit job
sbatch [sbatch file]

# Check job status
squeue -u [net id] # show all jobs
scancel -u [net id] -t [Job status:e.g. PD] # cancel jobs with specific status
scontrol show job [job id] # show job details
```

## Shared Data Folder

`/scratch/cl6707/Shared_Datasets/NSD_Mindeye`
