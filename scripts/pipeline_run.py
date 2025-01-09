# Import Libraries
import subprocess
import sys
import time

def run_pipeline():
    scripts = [
        "data_collection.py",
        "data_quality_check.py",
        "predictive_model.py",
        "predict.py",
        "live_predict.py"
    ]

    for script in scripts:
        print(f"\n{'=' * 20} Running {script} {'=' * 20}")
        result = subprocess.run([sys.executable, script], capture_output=True, text=True)
        print(result.stdout)
        if result.returncode != 0:
            print(f"Error in {script}:\n{result.stderr}")
            break
        time.sleep(2)

if __name__ == "__main__":
    run_pipeline()
