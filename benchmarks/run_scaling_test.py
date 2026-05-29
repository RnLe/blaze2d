import subprocess
import time
import csv

# Configuration
BINARY_PATH = "target/release/examples/scaling_benchmark"
OUTPUT_CSV = "scaling_benchmark_results.csv"
MAX_TIME = 300.0  # 5 minutes
STEPS = 1000
PHYSICAL_CORES = 16  # User specified
START_SIZE = 512

def compile_binary():
    print("Compiling benchmark binary...")
    subprocess.run(
        ["cargo", "build", "--release", "--example", "scaling_benchmark", "--features", "parallel", "-p", "blaze2d-core"],
        check=True
    )

def run_benchmark(size, threads):
    cmd = [
        f"./{BINARY_PATH}",
        "--size", str(size),
        "--threads", str(threads),
        "--steps", str(STEPS)
    ]
    
    try:
        # Run process and capture output
        # We don't use timeout argument here because we want to capture output even if it's slow,
        # but the logic adds check before running.
        # Actually, we should enforce a hard timeout on the process too just in case.
        start_t = time.time()
        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True,
            timeout=MAX_TIME + 30 
        )
        end_t = time.time()
        
        # Parse output
        total_time = None
        for line in result.stdout.splitlines():
            if line.startswith("Time:"):
                total_time = float(line.split(":")[1].strip())
        
        if total_time is None:
            print(f"Error parsing output for N={size}, T={threads}")
            print(result.stderr)
            return None
            
        print(f"  -> Time: {total_time:.4f}s")
        return total_time

    except subprocess.TimeoutExpired:
        print(f"  -> Timed out (> {MAX_TIME}s)")
        return None
    except Exception as e:
        print(f"  -> Error: {e}")
        return None

def main():
    compile_binary()
    
    results = []
    
    # Thread counts to test: 1 (Serial), then powers of 2 up to physical limit
    thread_counts = [1]
    c = 2
    while c <= PHYSICAL_CORES:
        thread_counts.append(c)
        c *= 2
        
    print(f"Testing scenarios: Threads={thread_counts}")

    for t in thread_counts:
        print(f"\nBenchmark Series: Threads = {t}")
        current_size = START_SIZE
        
        while True:
            print(f"Running Grid={current_size}x{current_size}...", end="", flush=True)
            
            elapsed = run_benchmark(current_size, t)
            
            if elapsed is not None:
                results.append({
                    "GridSize": current_size,
                    "Threads": t,
                    "Time": elapsed
                })
                
                # Check abortion conditions
                if elapsed > MAX_TIME:
                    print("  -> aborting series (exceeded 5m)")
                    break
                
                # Check prediction for next size (2x grid dimension = 4x elements)
                predicted_next = elapsed * 4.0
                if predicted_next > MAX_TIME:
                    print(f"  -> aborting series (next run predicted ~{predicted_next:.1f}s > {MAX_TIME}s)")
                    break
                
                current_size *= 2
            else:
                # Error or timeout
                break

    # Save Results
    print(f"\nSaving results to {OUTPUT_CSV}...")
    with open(OUTPUT_CSV, mode='w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["GridSize", "Threads", "Time"])
        writer.writeheader()
        for row in results:
            writer.writerow(row)
            
    # Also print summary table
    print("\nSummary:")
    print(f"{'Size':<10} {'Threads':<10} {'Time':<10}")
    for row in results:
         print(f"{row['GridSize']:<10} {row['Threads']:<10} {row['Time']:.4f}")

if __name__ == "__main__":
    main()
