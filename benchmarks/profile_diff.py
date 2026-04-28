#!/usr/bin/env python3
"""
Compare two profiling JSON reports and display the differences.

Usage:
    python profile_diff.py baseline.json current.json
    python profile_diff.py baseline.json current.json --threshold 5.0
    python profile_diff.py baseline.json current.json --append report.txt --label "TM mode"

The tool shows:
- Time difference (absolute and percentage) for each method
- Highlights significant regressions (>5% by default) in red
- Highlights significant improvements in green
- Summary statistics at the end

When using --append, reports are added to a growing log file for tracking changes over time.
"""

import json
import sys
import argparse
from datetime import datetime
from typing import Dict, Tuple, Optional, TextIO, List
from pathlib import Path

# ANSI color codes
RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
BOLD = "\033[1m"
RESET = "\033[0m"


def load_profile(path: str) -> Dict:
    """Load a profile JSON file."""
    with open(path, 'r') as f:
        return json.load(f)


def format_time(ms: float) -> str:
    """Format time with appropriate units."""
    if ms >= 1000:
        return f"{ms/1000:.3f}s"
    elif ms >= 1:
        return f"{ms:.2f}ms"
    else:
        return f"{ms*1000:.1f}µs"


def format_delta_plain(delta_ms: float, delta_pct: float) -> Tuple[str, str]:
    """Format delta without colors (for file output)."""
    sign = "+" if delta_ms > 0 else ""
    pct_sign = "+" if delta_pct > 0 else ""
    time_str = f"{sign}{format_time(delta_ms)}"
    pct_str = f"{pct_sign}{delta_pct:.1f}%"
    return time_str, pct_str


def format_delta(delta_ms: float, delta_pct: float, threshold: float) -> Tuple[str, str]:
    """Format delta with color based on threshold."""
    sign = "+" if delta_ms > 0 else ""
    pct_sign = "+" if delta_pct > 0 else ""
    
    if abs(delta_pct) >= threshold:
        if delta_pct > 0:
            color = RED  # Regression
        else:
            color = GREEN  # Improvement
    else:
        color = ""
    
    time_str = f"{sign}{format_time(delta_ms)}"
    pct_str = f"{pct_sign}{delta_pct:.1f}%"
    
    if color:
        return f"{color}{time_str}{RESET}", f"{color}{pct_str}{RESET}"
    return time_str, pct_str


def generate_report_lines(
    baseline: Dict, 
    current: Dict, 
    threshold: float,
    label: str = "",
    use_colors: bool = True
) -> Tuple[List[str], List[Tuple[str, float, float]], List[Tuple[str, float, float]]]:
    """
    Generate report lines and return (lines, regressions, improvements).
    """
    lines = []
    
    # Use color codes or empty strings
    r, g, y, c, b, rst = (RED, GREEN, YELLOW, CYAN, BOLD, RESET) if use_colors else ("", "", "", "", "", "")
    
    # Session totals
    base_total = baseline.get('session_total_ms', 0)
    curr_total = current.get('session_total_ms', 0)
    base_std = baseline.get('session_std_ms', 0)
    curr_std = current.get('session_std_ms', 0)
    base_iter = baseline.get('iterations', 1)
    curr_iter = current.get('iterations', 1)
    total_delta = curr_total - base_total
    total_pct = (total_delta / base_total * 100) if base_total > 0 else 0
    
    lines.append(f"{b}Session Total:{rst}")
    if base_std > 0:
        base_var_pct = (base_std / base_total * 100) if base_total > 0 else 0
        lines.append(f"  Baseline: {format_time(base_total)} ± {format_time(base_std)} ({base_var_pct:.1f}% var, {base_iter} runs)")
    else:
        lines.append(f"  Baseline: {format_time(base_total)}")
    if curr_std > 0:
        curr_var_pct = (curr_std / curr_total * 100) if curr_total > 0 else 0
        lines.append(f"  Current:  {format_time(curr_total)} ± {format_time(curr_std)} ({curr_var_pct:.1f}% var, {curr_iter} runs)")
    else:
        lines.append(f"  Current:  {format_time(curr_total)}")
    if use_colors:
        delta_str, pct_str = format_delta(total_delta, total_pct, threshold)
    else:
        delta_str, pct_str = format_delta_plain(total_delta, total_pct)
    lines.append(f"  Delta:    {delta_str} ({pct_str})")
    lines.append("")
    
    # Gather all method names
    base_methods = baseline.get('methods', {})
    curr_methods = current.get('methods', {})
    all_methods = set(base_methods.keys()) | set(curr_methods.keys())
    
    # Sort by current total time descending
    def sort_key(name):
        if name in curr_methods:
            return -curr_methods[name].get('total_ms', 0)
        elif name in base_methods:
            return -base_methods[name].get('total_ms', 0)
        return 0
    
    sorted_methods = sorted(all_methods, key=sort_key)
    
    # Header
    lines.append(f"{b}{'Method':<35} {'Base':>10} {'Current':>10} {'Delta':>12} {'Δ%':>8} {'Calls':>8}{rst}")
    lines.append("─" * 90)
    
    regressions = []
    improvements = []
    
    for name in sorted_methods:
        base_data = base_methods.get(name, {})
        curr_data = curr_methods.get(name, {})
        
        base_ms = base_data.get('total_ms', 0)
        curr_ms = curr_data.get('total_ms', 0)
        base_calls = base_data.get('calls', 0)
        curr_calls = curr_data.get('calls', 0)
        
        delta_ms = curr_ms - base_ms
        delta_pct = (delta_ms / base_ms * 100) if base_ms > 0 else (100 if curr_ms > 0 else 0)
        
        # Track significant changes
        if abs(delta_pct) >= threshold:
            if delta_pct > 0:
                regressions.append((name, delta_ms, delta_pct))
            else:
                improvements.append((name, delta_ms, delta_pct))
        
        # Format calls difference
        if base_calls != curr_calls:
            calls_str = f"{base_calls}→{curr_calls}"
        else:
            calls_str = str(curr_calls)
        
        # Format with colors
        if use_colors:
            delta_str, pct_str = format_delta(delta_ms, delta_pct, threshold)
        else:
            delta_str, pct_str = format_delta_plain(delta_ms, delta_pct)
        
        # Mark new/removed methods
        name_display = name[:35]
        if name not in base_methods:
            name_display = f"{c}+ {name_display}{rst}"
        elif name not in curr_methods:
            name_display = f"{y}- {name_display}{rst}"
        
        lines.append(f"{name_display:<35} {format_time(base_ms):>10} {format_time(curr_ms):>10} {delta_str:>20} {pct_str:>15} {calls_str:>8}")
    
    # Summary
    lines.append("")
    lines.append("─" * 90)
    
    if regressions:
        lines.append(f"\n{r}{b}Regressions (>{threshold}% slower):{rst}")
        for name, delta_ms, delta_pct in sorted(regressions, key=lambda x: -x[2]):
            marker = f"{r}▲{rst}" if use_colors else "▲"
            lines.append(f"  {marker} {name}: +{format_time(delta_ms)} (+{delta_pct:.1f}%)")
    
    if improvements:
        lines.append(f"\n{g}{b}Improvements (>{threshold}% faster):{rst}")
        for name, delta_ms, delta_pct in sorted(improvements, key=lambda x: x[2]):
            marker = f"{g}▼{rst}" if use_colors else "▼"
            lines.append(f"  {marker} {name}: {format_time(delta_ms)} ({delta_pct:.1f}%)")
    
    if not regressions and not improvements:
        lines.append(f"\n{g}No significant changes (threshold: ±{threshold}%){rst}")
    
    lines.append("")
    
    return lines, regressions, improvements


def compare_profiles(baseline: Dict, current: Dict, threshold: float = 5.0, label: str = "") -> int:
    """Compare two profiles and print the differences to stdout."""
    
    print(f"\n{BOLD}╭────────────────────────────────────────────────────────────────────────────────────────╮{RESET}")
    print(f"{BOLD}│                                  PROFILE COMPARISON                                     │{RESET}")
    print(f"{BOLD}╰────────────────────────────────────────────────────────────────────────────────────────╯{RESET}\n")
    
    if label:
        print(f"{BOLD}Label:{RESET} {label}\n")
    
    lines, regressions, improvements = generate_report_lines(baseline, current, threshold, label, use_colors=True)
    for line in lines:
        print(line)
    
    return 1 if regressions else 0


def append_report(
    baseline: Dict, 
    current: Dict, 
    report_path: str, 
    threshold: float = 5.0,
    label: str = "",
    baseline_path: str = "",
    current_path: str = ""
) -> int:
    """Append a comparison report to a file."""
    
    lines, regressions, improvements = generate_report_lines(baseline, current, threshold, label, use_colors=False)
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Check if file exists to determine if we need a header
    file_exists = Path(report_path).exists()
    
    with open(report_path, 'a') as f:
        # Separator for new entry
        f.write("\n")
        f.write("=" * 90 + "\n")
        f.write(f"                                  PROFILE COMPARISON\n")
        f.write("=" * 90 + "\n")
        f.write(f"Timestamp:   {timestamp}\n")
        if label:
            f.write(f"Label:       {label}\n")
        if baseline_path:
            f.write(f"Baseline:    {baseline_path}\n")
        if current_path:
            f.write(f"Current:     {current_path}\n")
        f.write(f"Description: \n")  # Left blank for manual entry
        f.write("-" * 90 + "\n")
        f.write("\n")
        
        for line in lines:
            f.write(line + "\n")
        
        f.write("\n")
    
    print(f"Report appended to: {report_path}")
    return 1 if regressions else 0


def main():
    parser = argparse.ArgumentParser(
        description='Compare two profiling JSON reports',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    %(prog)s baseline.json current.json
    %(prog)s baseline.json current.json --threshold 10.0
    %(prog)s baseline.json current.json --append report.txt --label "TM mode"
    %(prog)s results/before.json results/after.json --no-color
        """
    )
    parser.add_argument('baseline', help='Path to baseline profile JSON')
    parser.add_argument('current', help='Path to current profile JSON')
    parser.add_argument('--threshold', '-t', type=float, default=5.0,
                        help='Percentage threshold for highlighting changes (default: 5.0)')
    parser.add_argument('--no-color', action='store_true',
                        help='Disable colored output')
    parser.add_argument('--append', '-a', metavar='FILE',
                        help='Append report to a file instead of printing to stdout')
    parser.add_argument('--label', '-l', default='',
                        help='Label for this comparison (e.g., "TM mode", "after optimization")')
    
    args = parser.parse_args()
    
    # Disable colors if requested or not a TTY
    if args.no_color or not sys.stdout.isatty():
        global RED, GREEN, YELLOW, CYAN, BOLD, RESET
        RED = GREEN = YELLOW = CYAN = BOLD = RESET = ""
    
    try:
        baseline = load_profile(args.baseline)
        current = load_profile(args.current)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}", file=sys.stderr)
        sys.exit(1)
    
    if args.append:
        exit_code = append_report(
            baseline, current, args.append, args.threshold, args.label,
            baseline_path=args.baseline, current_path=args.current
        )
        # Also print to stdout
        compare_profiles(baseline, current, args.threshold, args.label)
    else:
        exit_code = compare_profiles(baseline, current, args.threshold, args.label)
    
    sys.exit(exit_code)


if __name__ == '__main__':
    main()
