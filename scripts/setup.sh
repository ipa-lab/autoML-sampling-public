#!/usr/bin/env bash
# this file was adapted from: https://github.com/josepablocam/ams/tree/master/experiments
echo "Installing resources for autoML-sampling"
source "scripts/folder_setup.sh"


if ! command -v conda > /dev/null
  then echo "Missing conda, please install"
  exit 1
fi
if [[ -z ${CONDA_EXE} ]]
then
    echo "Need to set CONDA_EXE environment variable"
    exit 1
fi

# install everything into appropriate conda environment
conda_folder=$(realpath "$(dirname $CONDA_EXE)/..")
source ${conda_folder}/etc/profile.d/conda.sh || { echo "Missing conda.sh"; exit 1; }

# Create base environment
conda activate autoML-sampling-env
if [[ $? -ne 0 ]]
then
    echo "Building conda environment autoML-sampling-env"
    conda env create -f environment.yml --force
    conda activate autoML-sampling-env
fi

conda activate autoML-sampling-env
#pip install xgboost

# install task-spooler
# https://vicerveza.homeunix.net/~viric/soft/ts/
if [[ $(uname) == "Darwin" ]]
then
    brew install task-spooler
else
    sudo apt-get install -y task-spooler
fi
