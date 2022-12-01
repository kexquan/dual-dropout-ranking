# Dual-dropout-ranking
This repository contains codes for dual-dropout-ranking (DDR).

Activate a new enviroment and install necessary packages:
pip install -r requirements.txt

Run example 1 (XOR dataset classification) in multithreading:
python DDR_main.py --run_example1 --operator_arch 128 32 4 --num_fs 3  --multi_thread

Run example 2 (Mnist hand-written digit feature importance visulization) in multithreading:
python DDR_main.py --run_example2 --operator_arch 128 32 2 --num_fs 50 --multi_thread

If you find this is useful, please cite "Dual Dropout Ranking of Linguistic Features for Alzheimer’s Disease Recognition" or "Automatic Selection of Spoken Language Biomarkers for Dementia Detection"
