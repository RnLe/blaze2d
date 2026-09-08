import os
import subprocess
import time

def run_benchmark():
    # N=2048, 50 Steps. Serial. Profiling.
    cmd = [
        "cargo", "run", "--release", 
        "-p", "blaze2d-core", 
        "--features", "parallel,profiling",
        "--example", "library_benchmark", 
        "--",
        "--resolution", "2048",
        "--steps", "50",
        "--silent"
    ]
    
    env = os.environ.copy()
    env["RAYON_NUM_THREADS"] = "1" # Force serial (though Parallel=False does it too usually)

    print("Running 2048x2048 Serial Benchmark (Profiling Enabled)...")
    start = time.time()
    
    # Run directly to see stdout (profiling stats)
    subprocess.run(cmd, env=env)
    
    end = time.time()
    print(f"Total Python Wall Time: {end - start:.4f}s")

if __name__ == "__main__":
    run_benchmark()
