import sys
import os
import re

def aws_goodone_mover():
    seedinfofilename = "G:/My Drive/research/nmsu/hatchery operation/codes/dynamic programming2/seed_info_temp.txt"
    awsip = ['18.217.98.180','3.23.55.216','3.128.53.228','3.150.79.75','3.17.54.22','3.19.172.165','3.150.238.61','3.150.195.20','3.133.53.235','3.128.193.159','3.12.170.116','18.118.89.246']

    # read the seed info file
    with open(seedinfofilename, 'r') as f:
        lines = f.readlines()
    collectdict = {}
    for line in lines:
        if 'aws' in line:
            # Extract the number after 'aws' and the number after 'zztrain_'
            # Example line: (10850)  # aws9\zztrain_120_10850.log
            # Method 1: Using regex
            aws_match = re.search(r'aws(\d+)', line)
            zztrain_match = re.search(r'zztrain_(\d+)', line)
            if aws_match and zztrain_match:
                awsid = int(aws_match.group(1))
                paramid = int(zztrain_match.group(1))
            else:
                # throw error
                raise ValueError(f"Line format incorrect: {line}")
            if f'aws{awsid}' not in collectdict:
                collectdict[f'aws{awsid}'] = {'paramid': paramid, 'seeds': [], 'ip': awsip[awsid-2]}
        else: # seed info
            seed_match = re.search(r'seed (\d+): ([\d.]+)', line.strip())
            if seed_match:
                seed_number = int(seed_match.group(1))
                collectdict[f'aws{awsid}']['seeds'].append(seed_number)
    
    # move the files to /TD3 results/goodones
    # bash command: scp -r -i "G:\My Drive\research\nmsu\aws keys\hatchery_test1.pem" "ubuntu@3.150.195.20:/home/ubuntu/Hatchery-operation-2/TD3 results/seed100639_paramset120" "G:\My Drive\research\nmsu\hatchery operation\codes\dynamic programming2\TD3 results\good_ones"
    for awskey in collectdict:
        paramid = collectdict[awskey]['paramid']
        seeds = collectdict[awskey]['seeds']
        ip = collectdict[awskey]['ip']
        for seed in seeds:
            sourcepath = f"ubuntu@{ip}:/home/ubuntu/Hatchery-operation-2/TD3 results/seed{seed}_paramset{paramid}"
            destpath = f"G:/My Drive/research/nmsu/hatchery operation/codes/dynamic programming2/TD3 results/good_ones"
            bashcommand = f'scp -r -q -i "G:/My Drive/research/nmsu/aws keys/hatchery_test1.pem" "{sourcepath}" "{destpath}"'
            print(f'Started moving seed {seed} from aws{awskey}...')
            os.system(bashcommand)
    print('All done!')

# execute 
if __name__ == "__main__":
    aws_goodone_mover()