import shutil
import sys
import os
# sys.argv[1] should be a spring representation of a list of seeds, e.g., "[1, 2, 3]"
seeds = eval(sys.argv[1])
algo = 'TD3'
paramset = 9
good_one_dir = f'./{algo} results/good_ones'
for seed in seeds:
    dirpath = f'./{algo} results/seed{seed}_paramset{paramset}'
    foldername = os.path.basename(dirpath) # extract just the folder name
    dest_path = f'{good_one_dir}/{foldername}'
    shutil.move(dirpath, dest_path)
    print(f"Moved {dirpath} → {dest_path}")