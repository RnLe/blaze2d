import os
import csv
import time
import sys
import subprocess

# Constants
OUTPUT_FILE = "benchmarks/results/results_meep_scaling.csv"
TIMEOUT = 60.0 
STEPS = 1000

os.makedirs("benchmarks/results", exist_ok=True)

# 1. Physical Cores
def get_physical_cores():
    try:
        result = subprocess.run("lscpu -p=Core,Socket | grep -v '^#' | sort -u | wc -l", shell=True, capture_output=True, text=True)
        return int(result.stdout.strip())
    except:
        return 16

PHYSICAL_CORES = get_physical_cores()

# 2. Param Space
GRID_SIZES = []
n = 512
while n <= 8192: 
    GRID_SIZES.append(n)
    n *= 2

THREAD_COUNTS = [1]
c = 2
while c <= PHYSICAL_CORES:
    THREAD_COUNTS.append(c)
    c *= 2

# Meep script content template
MEEP_SCRIPT = """
import meep as mp
import time
import sys

try:
    N = {N}
    STEPS = {STEPS}
    
    # 1.0 size domain. Resolution = N. 
    # This creates an NxN grid.
    
    cell_size = mp.Vector3(1, 1, 0)
    resolution = N 
    
    # Empty geometry (air) is fastest setup, pure FDTD benchmark
    # geometry = []
    
    # To match Rust benchmark (which uses eps=12 background with hole), 
    # we should use a block. 
    # Rust uses single_air_hole in a lattice. 
    # Let's just use a uniform block of eps=12 to force material updates.
    geometry = [mp.Block(mp.Vector3(mp.inf,mp.inf,mp.inf), material=mp.Medium(epsilon=12.0))]
    
    sim = mp.Simulation(
        cell_size=cell_size,
        resolution=resolution,
        geometry=geometry,
        boundary_layers=[mp.PML(0.1)], # Minimal PML
        # Courant is set on the class or defaults. 
        # Standard Meep defaults to 0.5 automatically.
    )
    
    # Init
    sim.init_sim()
    sim.reset_meep()
    
    # Calculate exact duration for STEPS
    # dt = 0.5 / resolution
    # duration = STEPS * dt
    dt = 0.5 / resolution
    bench_time = STEPS * dt

    # Force a single step first to warm up JIT/caches
    sim.run(until=1.0*dt)
    sim.reset_meep()
    
    start_t = time.time()
    sim.run(until=bench_time)
    end_t = time.time()
    
    print("BENCHMARK_TIME: " + str(end_t - start_t))

except Exception as e:
    # Print the exception to stderr so we catch it
    sys.stderr.write("MEEP SCRIPT ERROR: " + str(e) + "\\n")
    sys.exit(1)
"""

def run_meep_benchmark(n_grid, num_threads):
    # Write temp script
    script_content = MEEP_SCRIPT.format(N=n_grid, STEPS=STEPS)
    with open("temp_meep_bench.py", "w") as f:
        f.write(script_content)
        
    env = os.environ.copy()
    
    # Construct command
    if num_threads > 1:
        # Parallel execution using mpirun
        # Also set OMP_NUM_THREADS=1 so we don't oversubscribe threads vs processes
        cmd = ["mpirun", "-np", str(num_threads), sys.executable, "temp_meep_bench.py"]
        env["OMP_NUM_THREADS"] = "1"
        env["MKL_NUM_THREADS"] = "1"
    else:
        # Serial execution
        cmd = [sys.executable, "temp_meep_bench.py"]
        env["OMP_NUM_THREADS"] = "1"
        env["MKL_NUM_THREADS"] = "1"
    
    print(f"Meep: N={n_grid}, Threads={num_threads} ... ", end='', flush=True)
    
    try:
        start_wall = time.time()
        # Meep might dump a lot of info to stdout.
        proc = subprocess.run(
            cmd,
            env=env,
            capture_output=True,
            text=True,
            timeout=None # Unlimited timeout as requested
        )
        
        if proc.returncode != 0:
            print("FAILED")
            print(f"Stderr: {proc.stderr[:1000]}") # Print more error
            return None
            
        lines = proc.stdout.strip().split('\n')
        
        # We look for BENCHMARK_TIME: val
        valid_time = None
        for line in lines:
            if "BENCHMARK_TIME:" in line:
                try:
                    val = float(line.split(":")[1].strip())
                    valid_time = val
                    break
                except:
                    pass
                
        if valid_time is None:
            print("No valid time found. Output tail:")
            print('\n'.join(lines[-10:]))
            return None
            
        end_wall = time.time()
        total_wall = end_wall - start_wall
        overhead = total_wall - valid_time
        print(f"{valid_time:.4f}s [Total: {total_wall:.4f}s, Setup: {overhead:.4f}s]")
        
        # if valid_time > TIMEOUT:
        #    return float('inf')
            
        return valid_time

    except subprocess.TimeoutExpired:
        print("TIMEOUT")
        return float('inf')
    except Exception as e:
        print(f"Error: {e}")
        return None
    finally:
        if os.path.exists("temp_meep_bench.py"):
            os.remove("temp_meep_bench.py")

# Main Loop
with open(OUTPUT_FILE, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["Threads", "GridSize", "Time"])
    
    for threads in THREAD_COUNTS:
        print(f"\n=== MEEP SERIES: {threads} THREADS ===")
        previous_time = 0.0
        
        for n in GRID_SIZES:
            if previous_time > 0 and (previous_time * 4.0) > TIMEOUT:
                print(f"Skipping N={n}")
                break
                
            elapsed = run_meep_benchmark(n, threads)
            
            if elapsed is None:
                break
            
            if elapsed == float('inf'):
                break
                
            writer.writerow([threads, n, elapsed])
            f.flush()
            previous_time = elapsed
            
print(f"Done. Results: {OUTPUT_FILE}")
