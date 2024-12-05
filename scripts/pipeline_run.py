import subprocess

def run_pipeline():
    scripts = [
        "data_collection.py",
        "predictive_model.py",
        "predict.py",
        "live_predict.py"
    ]
    for script in scripts:
        print(f"Running {script}...")
        result = subprocess.run(["python", script], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"{script} executed successfully:\n{result.stdout}")
        else:
            print(f"Error in {script}:\n{result.stderr}")
            break

if __name__ == "__main__":
    run_pipeline()
