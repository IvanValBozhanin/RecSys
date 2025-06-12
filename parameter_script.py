import subprocess
import time
import re
import csv
import os
import sys

cov_types = ["standard", "RCV", "ACV", "hard_thr", "soft_thr"]
tau_values = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 1.5, 2, 5, 10, 50]
p_values = [0.05, 0.1, 0.15, 0.25, 0.45, 0.5, 0.55, 0.7, 0.75, 0.9]
output_file = (sys.argv[1] if len(sys.argv) > 1 else "sparsification_results") + ".csv"

results = []

for cov in cov_types:
    # Define parameter combinations based on covariance type
    if cov == "standard":
        # Standard has no parameters
        param_combos = [{"tau": None, "p": None}]
    elif cov in ["hard_thr", "soft_thr"]:
        # These use threshold only
        param_combos = [{"tau": tau, "p": None} for tau in tau_values]
    elif cov in ["RCV"]:
        # These use p only
        param_combos = [{"tau": None, "p": p} for p in p_values]
    elif cov in ["ACV"]:
        param_combos = [{"tau": None, "p": None}]
    
    # Run each parameter combination
    for params in param_combos:
        args = ["python", "train.py", "--cov_type", cov]
        
        if params["tau"] is not None:
            args += ["--tau", str(params["tau"])]
        if params["p"] is not None:
            args += ["--p", str(params["p"])]
            
        print(f"Running: {' '.join(args)}")
        start = time.time()
        result = subprocess.run(args, capture_output=True, text=True)
        end = time.time()
        
        stdout = result.stdout + result.stderr
        test_rmse = re.search(r"test_rmse\s+([0-9.]+)", stdout)
        sparsity = re.search(r"sparsity:\s*([0-9.]+)", stdout)  

        results.append({
            "cov_type": cov,
            "tau": params["tau"] if params["tau"] is not None else "",
            "p": params["p"] if params["p"] is not None else "",
            "test_rmse": float(test_rmse.group(1)) if test_rmse else None,
            "sparsity": float(sparsity.group(1)) if sparsity else None,
            "training_time": round(end - start, 2)
        })


with open(output_file, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["cov_type", "tau", "p", "test_rmse", "sparsity", "training_time"])
    writer.writeheader()
    writer.writerows(results)

print("All experiments completed and results saved to ", output_file)
