#!/bin/bash
# Append benchmark entry to YAML file
# Usage: ./append_benchmark.sh <yaml_file> <polarization> <log_output>
#
# Parses the bandstructure complete line from blaze output and appends to YAML.

YAML_FILE="$1"
POLARIZATION="$2"
LOG_FILE="$3"

# Fixed header values (from compare_square_lowres config)
LATTICE="square"
EPS_BG="13.0"
HOLE_RADIUS="0.3"
NUM_BANDS="8"
K_POINTS="61"  # 6 legs * 10 segments + 1
RESOLUTION="24"

# Create header if file doesn't exist
if [ ! -f "$YAML_FILE" ]; then
    cat > "$YAML_FILE" << EOF
# Benchmark tracking for blaze solver optimization
# Square lattice, ε_bg=$EPS_BG, r=$HOLE_RADIUS, $NUM_BANDS bands, $K_POINTS k-points, ${RESOLUTION}x${RESOLUTION} grid

runs:
EOF
fi

# Parse the bandstructure complete line
# Format: [bandstructure] complete: 61 k-points, 2045 total iterations, 3.52s (1.7ms/iter)
COMPLETE_LINE=$(grep "bandstructure.*complete" "$LOG_FILE" | tail -1)

if [ -z "$COMPLETE_LINE" ]; then
    echo "Error: Could not find bandstructure complete line in $LOG_FILE"
    exit 1
fi

# Extract values using sed
TOTAL_ITERS=$(echo "$COMPLETE_LINE" | sed -n 's/.*complete: [0-9]* k-points, \([0-9]*\) total iterations.*/\1/p')
TOTAL_TIME_RAW=$(echo "$COMPLETE_LINE" | sed -n 's/.*, \([0-9.]*\)s (.*/\1/p')
TIME_PER_ITER_RAW=$(echo "$COMPLETE_LINE" | sed -n 's/.*(\([0-9.]*\)ms\/iter).*/\1/p')

# Format times to 3 decimal places
TOTAL_TIME=$(printf "%.3f" "$TOTAL_TIME_RAW")
TIME_PER_ITER=$(printf "%.3f" "$TIME_PER_ITER_RAW")

# Extract min/max iterations from eigensolver lines
MIN_ITER=$(grep "eigensolver.*iters=" "$LOG_FILE" | sed -n 's/.*iters=[ ]*\([0-9]*\).*/\1/p' | sort -n | head -1)
MAX_ITER=$(grep "eigensolver.*iters=" "$LOG_FILE" | sed -n 's/.*iters=[ ]*\([0-9]*\).*/\1/p' | sort -n | tail -1)

# Get timestamp
TIMESTAMP=$(date -Iseconds)

# Append entry
cat >> "$YAML_FILE" << EOF
  - timestamp: "$TIMESTAMP"
    polarization: "$POLARIZATION"
    min_iter: $MIN_ITER
    max_iter: $MAX_ITER
    total_iterations: $TOTAL_ITERS
    total_time_s: $TOTAL_TIME
    time_per_iter_ms: $TIME_PER_ITER
    description: ""
EOF

echo "Appended $POLARIZATION benchmark entry to $YAML_FILE"
