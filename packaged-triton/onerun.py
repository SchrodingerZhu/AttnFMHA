
import subprocess
import re
import os
import sys
import json
import matplotlib.pyplot as plt
import numpy as np
import argparse

# --- Configurations ---
VARIANTS = ['default', 'tile', 'tile_alt', 'fully_static', 'fully_static_alt']
CAUSAL_OPTIONS = [False, True]
QUANT_OPTIONS = ['fp16', 'fp8_e4m3'] 

PYTHON_EXEC = sys.executable
ENTRY_SCRIPT = "AttentionFMHAEntryPoint.py"
NCU_PATH = "/opt/nvidia/nsight-compute/2025.3.1/ncu"

# Metric for L2 Misses (adjust if needed specific to architecture, user used this)
METRIC_NAME = "lts__t_sectors_srcunit_tex_lookup_miss.sum"

def run_command(cmd, capture=True):
    """Run a subprocess command and return output."""
    try:
        if capture:
            result = subprocess.run(cmd, capture_output=True, text=True)
            return result.returncode, result.stdout, result.stderr
        else:
            result = subprocess.run(cmd)
            return result.returncode, "", ""
    except Exception as e:
        return -1, "", str(e)

def parse_ncu_misses(output):
    """Parse NCU CSV output for the miss count."""
    # Pattern: "lts__...", "sector", "123,456"
    values = []
    # Flexible regex for CSV lines
    pattern = re.compile(rf'"{METRIC_NAME}","[^"]*","?([\d,\.]+)"?')
    
    for line in output.splitlines():
        if METRIC_NAME in line:
            # simple split approach first if regex fails or complicates
            # But let's try regex
            m = pattern.search(line)
            if m:
                val_str = m.group(1).replace(',', '')
                try:
                    values.append(float(val_str))
                except ValueError:
                    pass
            else:
                # Fallback: simple split
                parts = line.split(',')
                # find the part with miss name, look ahead for value
                # This is brittle, rely on the fact we grepped the metric name
                pass
    
    if values:
        # Return max value (typically the main kernel)
        return max(values)
    return 0.0

def parse_tflops(output):
    """Parse TFlops from script output."""
    match = re.search(r"Estimated TFlops/sec:\s*([\d\.]+)", output)
    if match:
        return float(match.group(1))
    return 0.0

def collect_data():
    data = []
    
    total_steps = len(VARIANTS) * len(CAUSAL_OPTIONS) * len(QUANT_OPTIONS)
    step = 0
    
    print(f"Starting collection: {total_steps} configurations.")

    for variant in VARIANTS:
        for causal in CAUSAL_OPTIONS:
            for quant in QUANT_OPTIONS:
                step += 1
                config_str = f"Variant={variant}, Causal={causal}, Quant={quant}"
                print(f"[{step}/{total_steps}] Running {config_str}...")
                
                # Setup Base Args
                base_args = [ENTRY_SCRIPT, "--variant", variant, "--quant", quant]
                # Note: AttentionFMHAEntryPoint might not have --causal flag exposed directly in main?
                # Checked file: It does NOT have --causal flag in argparse!
                # It hardcodes CAUSAL=False inside run_benchmark!
                # I need to FIX AttentionFMHAEntryPoint.py to accept --causal.
                # BUT wait, I should check if I can modify it.
                # Assuming I will modify EntryPoint to take --causal argument.
                # Adding that to the plan/execution now.
                
                # For now assume I added it.
                if causal:
                    base_args.append("--causal")
                
                # 1. Profile Misses
                cmd_profile = ["sudo", "-E", NCU_PATH, "--metric", METRIC_NAME, "--csv", PYTHON_EXEC] + base_args
                
                misses = 0.0
                # Only run profiling if we can (sudo might be needed)
                # If sudo fails or not present, we skip or mock? User asked for it similar to study5.
                rc, out, err = run_command(cmd_profile)
                if rc == 0:
                    misses = parse_ncu_misses(out)
                else:
                    print(f"  Profiling failed: {err[:100]}...")
                
                # 2. Benchmark TFlops (Run normally)
                cmd_bench = [PYTHON_EXEC] + base_args
                rc, out, err = run_command(cmd_bench)
                tflops = 0.0
                if rc == 0:
                    tflops = parse_tflops(out)
                else:
                    print(f"  Bench failed: {err[:100]}...")

                print(f"  -> Misses: {misses:,.0f}, TFlops: {tflops:.2f}")
                
                record = {
                    "variant": variant,
                    "causal": causal,
                    "quant": quant,
                    "misses": misses,
                    "tflops": tflops
                }
                data.append(record)
    
    return data

def plot_graph(data):
    # We want to plot TFlops and Misses.
    # Group by (Causal, Quant) -> Comparison of Variants
    
    # Filter unique configs
    configs = set((d['causal'], d['quant']) for d in data)
    
    for (causal, quant) in configs:
        subset = [d for d in data if d['causal'] == causal and d['quant'] == quant]
        # Sort by variant order
        subset.sort(key=lambda x: VARIANTS.index(x['variant']))
        
        labels = [d['variant'] for d in subset]
        misses = [d['misses'] for d in subset]
        tflops = [d['tflops'] for d in subset]
        
        x = np.arange(len(labels))
        width = 0.35
        
        fig, ax1 = plt.subplots(figsize=(10, 6))
        
        # Bar 1: TFlops
        color = 'tab:blue'
        ax1.set_xlabel('Variant')
        ax1.set_ylabel('TFlops', color=color)
        bars1 = ax1.bar(x - width/2, tflops, width, color=color, label='TFlops')
        ax1.tick_params(axis='y', labelcolor=color)
        
        # Bar 2: Misses
        ax2 = ax1.twinx()
        color = 'tab:red'
        ax2.set_ylabel('L2 Misses', color=color)
        bars2 = ax2.bar(x + width/2, misses, width, color=color, label='L2 Misses')
        ax2.tick_params(axis='y', labelcolor=color)
        
        # Title
        causal_str = "Causal" if causal else "Non-Causal"
        plt.title(f"FMHA Performance: {causal_str}, {quant}")
        
        ax1.set_xticks(x)
        ax1.set_xticklabels(labels, rotation=45, ha='right')
        
        # Layout
        fig.tight_layout()
        filename = f"plot_{'causal' if causal else 'noncausal'}_{quant}.png"
        plt.savefig(filename)
        print(f"Saved plot: {filename}")

if __name__ == "__main__":
    # Check if we need to modify EntryPoint for Causal support first
    # This script assumes EntryPoint has --causal
    
    data = collect_data()
    
    # Save raw data
    with open("results.json", "w") as f:
        json.dump(data, f, indent=2)
        
    plot_graph(data)
