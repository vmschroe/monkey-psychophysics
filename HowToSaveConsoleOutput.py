#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 11 18:17:00 2025

@author: vmschroe
"""

import sys
import os
import pickle
from datetime import datetime
import subprocess

# Create a timestamp for the output files
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = "testISH_output_" + timestamp
os.makedirs(output_dir, exist_ok=True)

# Set up logging
log_path = os.path.join(output_dir, "testISH_log.txt")
log_file = open(log_path, "w")
original_stdout = sys.stdout
sys.stdout = log_file

try:
    print(f"testISH started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    ######## Run your testISH and analysis
    psych_vecs_sim, Ls = [[12345, 98765], [666, 444]]
    trace = [1,2,3,4,5,6]
    
    # Save the results
    results_path = os.path.join(output_dir, "testISH_results.pkl")
    with open(results_path, 'wb') as f:
        pickle.dump({
            'psych_vecs_sim': psych_vecs_sim,
            'Ls': Ls,
            'trace': trace
        }, f)
    
    print(f"Results saved to {results_path}")
    print(f"testISH completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

except Exception as e:
    print(f"Error occurred: {e}")
    import traceback
    traceback.print_exc(file=log_file)

finally:
    # Restore original stdout
    sys.stdout = original_stdout
    log_file.close()
    print(f"testISH complete. Log saved to {log_path}")
    
    # Add the new files to git
    subprocess.run(["git", "add", output_dir], check=True)
    
    # Use your gpush command to commit and push to GitHub
    try:
        # Stage new files
        subprocess.run(["git", "add", output_dir], check=True)
        
        # Commit changes
        commit_message = f"Add simulation results from {timestamp}"
        subprocess.run(["git", "commit", "-m", commit_message], check=True)
        
        # Push to remote repository
        subprocess.run(["git", "push"], check=True)
        
        print("Successfully pushed results to GitHub")
    except subprocess.CalledProcessError as e:
        print(f"Failed to push to GitHub: {e}")