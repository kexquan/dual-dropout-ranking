# Dual-dropout-ranking
This repository contains codes for dual-dropout-ranking (DDR). The codes only work on GPUs.

## Prepare Environment
Activate a new enviroment and install necessary packages:

pip install -r requirements.txt

Tips: If there is any missing package, please refer to 'requirements_full.txt' to install corresponding packages.

## Example 1: XOR dataset classification
Run example 1 in multithreading, which will take about 5 minutes on a RTX3090 GPU:

python DDR_main.py --run_example1 --operator_arch 128 32 4 --num_fs 3  --multi_thread

## Example 2: MNIST hand-written digit feature importance visulization
Run example 2 in multithreading, which will take about 5 minutes on a RTX3090 GPU:

python DDR_main.py --run_example2 --operator_arch 128 32 2 --num_fs 50 --multi_thread

##
If you find this is useful, please cite 
[Dual Dropout Ranking of Linguistic Features for Alzheimer’s Disease Recognition](http://www.eie.polyu.edu.hk/~mwmak/papers/apsipa21b.pdf)
and Automatic Selection of Spoken Language Biomarkers for Dementia Detection.

##
Homepage: <https://kexquan.github.io>

Email: xiaoquan.ke@connect.polyu.hk
