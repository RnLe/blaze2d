import os
import subprocess
import time
import multiprocessing

def get_physical_cores():
    # Attempt to get physical core count
    # Default to 1 if detection fails to be safe, but usually this works
    try:
        import psutil
        return psutil.cpu_count(logical=False) or multiprocessing.cpu_count()
    except ImportError:
        return multiprocessing.cpu_count()

def run_benchmark_iteration(size, parallel, threads=None):
    cmd = [
        "cargo", "run", "--release", 
        "-p", "blaze2d-core", 
        "--features", "parallel,profiling",
        "--example", "library_benchmark", 
        "--",
        "--resolution", str(size),
        "--steps", "50",  # 50 Steps for benchmark
        "--silent"
    ]
    
    if parallel:
        cmd.append("--parallel")
    
    env = os.environ.copy()
    if threads:
        env["RAYON_NUM_THREADS"] = str(threads)
    else:
        # Determine for serial run or just robust
        env["RAYON_NUM_THREADS"] = "1"

    print(f"Running: Size={size}, Parallel={parallel}, Threads={threads}...")
    
    start_time = time.time()
    try:
        # Timeout at 300 seconds (5 minutes)
        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True, 
            env=env,
            timeout=300 
        )
        if result.returncode != 0:
            print(f"Error: {result.stderr}")
            return None
        
        # Parse output (last line should be time in seconds)
        lines = result.stdout.strip().split('\n')
        # Filter for the number
        duration_str = lines[-1].strip()
        try:
             duration = float(duration_str)
        except ValueError:
             print(f"Could not parse duration: {duration_str}")
             return None
             
        return duration
        
    except subprocess.TimeoutExpired:
        print(f"Timeout expired for Size={size}, Threads={threads}")
        return "TIMEOUT"

def main():
    output_file = "benchmark_scaling_results.csv"
    physical_cores = get_physical_cores()
    max_cores = physical_cores
    
    # Grid sizes: start at 512, double until ...
    # N=512, 1024, 2048, 4096, 8192
    grid_powers = [2**i for i in range(9, 14)] # 512 to 8192
    
    # Thread counts: 2, 4, 8, ... up to max_cores
    thread_counts = []
    c = 2
    while c <= max_cores:
        thread_counts.append(c)
        c *= 2
        
    print(f"Physical Cores Detected: {physical_cores}")
    print(f"Testing Grid Sizes: {grid_powers}")
    print(f"Testing Thread Counts: {thread_counts}")

    results = []

    # 1. Serial Run (Single Core)
    print("\n--- Starting Serial Benchnarks ---")
    for size in grid_powers:
        duration = run_benchmark_iteration(size, False, 1)
        results.append({
            "GridSize": size,
            "Mode": "Serial", 
            "Threads": 1,
            "Time": duration
        })
        if duration == "TIMEOUT":
            print("Aborting remaining serial runs due to timeout.")
            break
            
    # 2. Parallel Runs
    print("\n--- Starting Parallel Benchnarks ---")
    
    # For Parallel, we iterate by thread count first? Or size first?
    # "Then do the SAME series for parallel, but with just 2 cores. Rerun... Go down in powers of 2 regarding core count"
    # Actually user said: "do the SAME series for parallel, but with just 2 cores. Rerun. Choose the same abortion parameter."
    # Then "Go down in powers of 2 regarding the core count until reaching the physical" (Wait, this phrasing is odd. Usually 'go up' to physical. 'Go down' suggests starting high?)
    # "Go down in powers of 2 ... until reaching physical" -> 2, 4, 8 ... might be 'going down the list'? 
    # Or start at physical and go down to 2?
    # "Do the SAME series for parallel... with just 2 cores." -> Start with 2.
    # "Go down in powers of 2... until reaching physical" -> likely means "Iterate through powers of 2: 2, 4, 8..."
    
    for threads in thread_counts:
        print(f"\n--- Testing with {threads} Threads ---")
        for size in grid_powers:
            duration = run_benchmark_iteration(size, True, threads)
            results.append({
                "GridSize": size,
                "Mode": "Parallel", 
                "Threads": threads,
                "Time": duration
            })
            if duration == "TIMEOUT":
                print(f"Aborting remaining runs for thread count {threads} due to timeout.")
                break

def main():
    print("Starting Blaze2D Scaling Benchmark (N=2048, Serial check)")
    run_benchmark_iteration(2048, False, 1)

if __name__ == "__main__":
    main()
