import subprocess
import os
import csv
import time
import sys

# Constants
OUTPUT_FILE = "benchmarks/results/results_rust_scaling.csv"
TIMEOUT = 60.0  # Constraint: 1 minute
STEPS = 1000

# Ensure results directory exists
os.makedirs("benchmarks/results", exist_ok=True)

# 1. Detect Physical Cores
def get_physical_cores():
    try:
        # lscpu -p=Core,Socket returns list of core/socket for each CPU
        result = subprocess.run(
            "lscpu -p=Core,Socket | grep -v '^#' | sort -u | wc -l", 
            shell=True, capture_output=True, text=True
        )
        count = int(result.stdout.strip())
        return count
    except Exception as e:
        print(f"Could not automatically detect cores via lscpu: {e}")
        return 16 # Fallback

PHYSICAL_CORES = get_physical_cores()
print(f"Detected Physical Cores: {PHYSICAL_CORES}")

# 2. Define Param Space
# Power of 2 grid sizes
GRID_SIZES = []
n = 512
while n <= 32768:
    GRID_SIZES.append(n)
    n *= 2

# Threads: 1 (Serial), then 2, 4... up to Physical
THREAD_COUNTS = [1]
c = 2
while c <= PHYSICAL_CORES:
    THREAD_COUNTS.append(c)
    c *= 2

# 3. Benchmark Runner
def run_benchmark(n_grid, num_threads, use_parallel_flag):
    cmd = [
        "cargo", "run", "--release", "-p", "blaze2d-core",
        "--features", "parallel",
        "--example", "library_benchmark",
        "--",
        "--resolution", str(float(n_grid)),
        "--steps", str(STEPS),
        "--silent"
    ]
    
    if use_parallel_flag:
        cmd.append("--parallel")
    
    env = os.environ.copy()
    if use_parallel_flag:
        env["RAYON_NUM_THREADS"] = str(num_threads)
        
    print(f"Running: N={n_grid}, Threads={num_threads}, Parallel={use_parallel_flag} ... ", end='', flush=True)
    
    try:
        start_wall = time.time()
        # Allow extra time for compilation overhead, but we check the logic time separately
        proc = subprocess.run(
            cmd, 
            env=env, 
            capture_output=True, 
            text=True, 
            timeout=300.0
        )
        
        if proc.returncode != 0:
            print(f"FAILED (RC {proc.returncode})")
            if "build" not in proc.stderr: # Only show error if not build noise
                 print(f"Stderr: {proc.stderr[:200]}...")
            return None
            
        lines = proc.stdout.strip().split('\n')
        if not lines:
             print("No output")
             return None
             
        target_line = lines[-1].strip()
        
        try:
            core_time = float(target_line)
            
            end_wall = time.time()
            total_wall = end_wall - start_wall
            overhead = total_wall - core_time
            print(f"{core_time:.4f}s [Total: {total_wall:.4f}s, Setup: {overhead:.4f}s]")
            
            # if core_time > TIMEOUT:
            #    return float('inf')
                
            return core_time
        except ValueError:
            print(f"Parse Error: '{target_line}'")
            return None

    except subprocess.TimeoutExpired:
        print("TIMEOUT (Wall)")
        return float('inf')
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        return None

# 4. Main Loop
with open(OUTPUT_FILE, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["Threads", "GridSize", "Time", "ParallelEnabled"])
    
    for threads in THREAD_COUNTS:
        is_parallel = (threads > 1)
        mode_str = "PARALLEL" if is_parallel else "SERIAL"
        print(f"\n=== SERIES: {threads} THREADS ({mode_str}) ===")
        
        previous_time = 0.0
        
        for n in GRID_SIZES:
            # Heuristic: Check if next run is likely to exceed timeout
            if previous_time > 0 and (previous_time * 4.0) > TIMEOUT:
                print(f"Skipping N={n} (projected > {TIMEOUT}s)")
                break
                
            elapsed = run_benchmark(n, threads, is_parallel)
            
            if elapsed is None:
                break
            
            if elapsed == float('inf') or elapsed > TIMEOUT:
                print(f" Aborting series (Time > {TIMEOUT}s)")
                break
            
            writer.writerow([threads, n, elapsed, is_parallel])
            f.flush()
            previous_time = elapsed

print(f"\nDone. Results: {OUTPUT_FILE}")
