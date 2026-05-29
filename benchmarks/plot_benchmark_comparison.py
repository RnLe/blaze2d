import pandas as pd
import matplotlib.pyplot as plt
import os

# Paths
RUST_RESULTS = "benchmarks/results/results_rust_scaling.csv"
MEEP_RESULTS = "benchmarks/results/results_meep_scaling.csv"
OUTPUT_PLOT = "benchmarks/results/comparison_plot.png"

def plot():
    if not os.path.exists(RUST_RESULTS):
        print(f"Missing {RUST_RESULTS}")
    
    if not os.path.exists(MEEP_RESULTS):
        print(f"Missing {MEEP_RESULTS}")

    plt.figure(figsize=(12, 10))
    
    # Plot Rust
    if os.path.exists(RUST_RESULTS):
        try:
            rust_df = pd.read_csv(RUST_RESULTS)
            rust_threads = sorted(rust_df['Threads'].unique())
            # Use cool colors for Rust
            cmap = plt.cm.Blues
            for i, t in enumerate(rust_threads):
                subset = rust_df[rust_df['Threads'] == t].sort_values('GridSize')
                color = cmap(0.5 + 0.5 * (i / len(rust_threads)))
                plt.loglog(subset['GridSize'], subset['Time'], marker='o', linestyle='-', color=color, label=f"Rust {t}T")
        except Exception as e:
            print(f"Error plotting Rust: {e}")

    # Plot Meep
    if os.path.exists(MEEP_RESULTS):
        try:
            meep_df = pd.read_csv(MEEP_RESULTS)
            meep_threads = sorted(meep_df['Threads'].unique())
            # Use warm colors for Meep
            cmap = plt.cm.Reds
            for i, t in enumerate(meep_threads):
                subset = meep_df[meep_df['Threads'] == t].sort_values('GridSize')
                color = cmap(0.5 + 0.5 * (i / len(meep_threads)))
                plt.loglog(subset['GridSize'], subset['Time'], marker='x', linestyle='--', color=color, label=f"Meep {t}T")
        except Exception as e:
            print(f"Error plotting Meep: {e}")

    plt.xlabel("Grid Size (N)")
    plt.ylabel("Time (s)")
    plt.title("Scaling Comparison: Meep vs Rust FDTD")
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.5)
    
    plt.savefig(OUTPUT_PLOT)
    print(f"Plot saved to {OUTPUT_PLOT}")

if __name__ == "__main__":
    plot()
