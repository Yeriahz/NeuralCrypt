import subprocess
import sys
import time

def run_pipeline():
    scripts = [
        "data_collection.py",
        "predictive_model.py",
        "predict.py",
        "live_predict.py"
    ]

    for script in scripts:
        print(f"\n{'=' * 20} Running {script} {'=' * 20}")
        
        try:
            result = subprocess.run([sys.executable, script], capture_output=True, text=True)
            if result.returncode == 0:
                print(f"{script} executed successfully:\n{result.stdout}")
            else:
                print(f"Error in {script}:\n{result.stderr}")
                break
        except KeyboardInterrupt:
            print("\nPipeline interrupted by user.")
            break
        except Exception as e:
            print(f"Unexpected error while running {script}: {e}")
            break

        # Wait for 2 seconds between scripts to allow for database transactions or logging completion
        time.sleep(2)

if __name__ == "__main__":
    run_pipeline()
