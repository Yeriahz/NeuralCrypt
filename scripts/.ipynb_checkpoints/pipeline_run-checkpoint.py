# Import Libraries
import subprocess
import sys
import time
from colorama import init, Fore, Style

init(autoreset=True)

def banner_print(message, color=Fore.CYAN):
    """Helper function to print a visually distinct banner."""
    print("\n" + "=" * 70)
    print(color + Style.BRIGHT + message + Style.RESET_ALL)
    print("=" * 70 + "\n")

def run_pipeline():
    """
    Runs each script in order, streaming output in real time.
    If any script fails, we stop the pipeline.
    """
    scripts = [
        "data_collection.py",
        "data_quality_check.py",
        "predictive_model.py",
        "predict.py",
        "live_predict.py"
    ]

    for i, script in enumerate(scripts):
        banner_print(f"Running {script}", Fore.MAGENTA)

        try:
            result = subprocess.run(
                [sys.executable, script],
                check=True,        # Raises CalledProcessError if script fails
                text=True,
                bufsize=1,         # Line-buffered
                stdout=sys.stdout, # Stream directly to console
                stderr=sys.stderr
            )
        except subprocess.CalledProcessError as e:
            print(Fore.RED + f"\nError in {script} (exit code {e.returncode}):" + Style.RESET_ALL)
            print(Fore.RED + f"{e}" + Style.RESET_ALL)
            print(Fore.RED + "Terminating pipeline." + Style.RESET_ALL)
            break

        # Optional small pause between scripts
        if i < len(scripts) - 1:
            time.sleep(2)

def main():
    banner_print("STARTING FULL PIPELINE", Fore.GREEN)
    run_pipeline()
    banner_print("PIPELINE FINISHED (or last script is still running)", Fore.GREEN)

if __name__ == "__main__":
    main()