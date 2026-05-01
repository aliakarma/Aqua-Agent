import argparse
import sys
import subprocess
import os

def main():
    print("=============================================")
    print(" AquaAgent Reproducibility Checker")
    print("=============================================")
    if not os.path.exists("data/raw/aquaagent_dma.inp"):
        print("CRITICAL: Real EPANET physical environment is missing.")
        print("Using random walk mock mode is strictly prohibited for NeurIPS reproducibility runs.")
        print("Please obtain and place the data in data/raw/aquaagent_dma.inp and install epyt.")
        sys.exit(1)

    print("Running training sequences (with seed variations)")
    subprocess.run(["bash", "scripts/run_train_ada.sh", "42", "cpu"], check=True)
    subprocess.run(["bash", "scripts/run_train_mappo.sh", "42", "cpu"], check=True)
    subprocess.run(["bash", "scripts/run_evaluate.sh", "cpu"], check=True)

if __name__ == "__main__":
    main()
