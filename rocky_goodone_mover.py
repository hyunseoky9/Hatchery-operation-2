import os
import re
import subprocess
from pathlib import Path


def goodone_mover_rocky():
    # -------------------- EDIT THESE --------------------
    seedinfofilename = "G:/My Drive/research/nmsu/hatchery operation/codes/dynamic programming2/seed_info_temp.txt"

    rocky_host = "rocky.nimbios.org"
    rocky_user = "hyoon15"

    # Your NEW OpenSSH/PEM key that you verified works with `ssh -i ...`
    pem_keyfile = "G:/My Drive/research/nmsu/rockyHPC/rocky_nopass.pem"

    # Remote base directory that contains the folders like:
    # seed{seed}_paramset{paramid}
    remote_base = "/home/hyoon15/Hatchery-operation-2/TD3 results"

    # Local destination folder
    local_dest = "G:/My Drive/research/nmsu/hatchery operation/codes/dynamic programming2/TD3 results/good_ones"
    # ----------------------------------------------------


    def escape_remote_path(p: str) -> str:
        """Escape spaces for scp remote path."""
        return p.replace(" ", r"\ ")

    # Read seed info file
    with open(seedinfofilename, "r") as f:
        lines = f.readlines()

    # Parse into jobs: [(paramid, [seed, seed, ...]), ...]
    jobs = []
    current_paramid = None
    current_seeds = []

    for line in lines:
        zztrain_match = re.search(r"zztrain_(\d+)", line)
        if zztrain_match:
            # flush previous block
            if current_paramid is not None and current_seeds:
                jobs.append((current_paramid, current_seeds))
            current_paramid = int(zztrain_match.group(1))
            current_seeds = []
            continue

        seed_match = re.search(r"seed\s+(\d+)\s*:", line.strip())
        if seed_match and current_paramid is not None:
            current_seeds.append(int(seed_match.group(1)))

    # flush last block
    if current_paramid is not None and current_seeds:
        jobs.append((current_paramid, current_seeds))

    if not jobs:
        raise ValueError(
            "No (paramid, seeds) blocks found. Make sure your seed_info_temp.txt contains lines with "
            "'zztrain_<paramid>' and subsequent 'seed <number>:' lines."
        )

    Path(local_dest).mkdir(parents=True, exist_ok=True)

    # Copy folders
    for paramid, seeds in jobs:
        for seed in seeds:
            remote_folder = f"{remote_base}/seed{seed}_paramset{paramid}"
            remote_spec = f"{rocky_user}@{rocky_host}:{remote_folder}"

            cmd = [
                "scp",
                "-r",
                "-q",
                "-i", pem_keyfile,
                "-o", "BatchMode=yes",          # fail fast if auth breaks (no prompts)
                remote_spec,
                local_dest,
            ]

            print(f"Copying seed {seed} (paramset {paramid}) from Rocky...")
            result = subprocess.run(cmd)

            if result.returncode != 0:
                print(f"Failed: seed {seed}, paramset {paramid} (exit {result.returncode})")

    print("All done!")


if __name__ == "__main__":
    goodone_mover_rocky()
