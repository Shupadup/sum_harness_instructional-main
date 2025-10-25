"""
E. Wes Bethel, Copyright (C) 2022

October 2022

Description: This code loads a .csv file and creates a 3-variable plot

Inputs: the named file "sample_data_3vars.csv"

Outputs: displays a chart with matplotlib

Dependencies: matplotlib, pandas modules

Assumptions: developed and tested using Python version 3.8.8 on macOS 11.6

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


fname = "perlmutter_all_methods_metrics.csv"
df = pd.read_csv(fname, comment="#")
print(df)

var_names = list(df.columns)

print("var names =", var_names)

# split the df into individual vars
# assumption: column order - 0=problem size, 1=blas time, 2=basic time
# UPDATED: the CSV is in *long* form with columns:
#   method (sum_direct/sum_indirect/sum_vector), N, time_s, MFLOP/s, %_peak_bw, avg_latency_ns, etc.

# Ensure required columns exist
required_cols = {"method", "N", "time_s", "MFLOP/s", "%_peak_bw", "avg_latency_ns"}
missing = required_cols.difference(df.columns)
if missing:
    raise ValueError(f"CSV is missing required columns: {missing}")

# Coerce numeric columns (handles blanks as NaN)
for col in ["N", "time_s", "MFLOP/s", "%_peak_bw", "avg_latency_ns"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# We’ll pivot to *wide* form per metric: index=N, columns=method, values=metric
def wide_for(metric_col: str) -> pd.DataFrame:
    w = (df[["N", "method", metric_col]]
         .dropna(subset=["N"])
         .pivot(index="N", columns="method", values=metric_col)
         .sort_index())
    # Reorder/ensure expected columns if present
    cols = [c for c in ["sum_direct", "sum_indirect", "sum_vector"] if c in w.columns]
    return w[cols]

def plot_metric(metric_col: str, ylabel: str, title: str, out_png: str):
    w = wide_for(metric_col)
    # X axis: problem sizes
    problem_sizes = w.index.to_list()
    xlocs = list(range(len(problem_sizes)))

    plt.figure()
    plt.title(title)

    plt.xticks(xlocs, problem_sizes, rotation=45)

    # Plot up to three series if present
    if "sum_direct" in w.columns:
        plt.plot(w["sum_direct"].to_list(), "r-o", label="sum_direct")
    if "sum_indirect" in w.columns:
        plt.plot(w["sum_indirect"].to_list(), "b-x", label="sum_indirect")
    if "sum_vector" in w.columns:
        plt.plot(w["sum_vector"].to_list(), "g-^", label="sum_vector")

    #plt.xscale("log")
    #plt.yscale("log")

    plt.xlabel("Problem Sizes (N)")
    plt.ylabel(ylabel)
    plt.legend(loc="best")
    plt.grid(axis="both")

    plt.tight_layout()
    # Save a PNG you can submit (you can also use the GUI to save as PNG/PDF)
    plt.savefig(out_png, dpi=200)
    plt.show()

# --- Produce the three required charts ---
plot_metric("MFLOP/s",
            ylabel="MFLOP/s",
            title="Problem Size vs. MFLOP/s",
            out_png="mflops_vs_problem_size.png")

plot_metric("%_peak_bw",
            ylabel="% of Peak Memory Bandwidth",
            title="Problem Size vs. % Peak Memory Bandwidth Utilized",
            out_png="bandwidth_pct_vs_problem_size.png")

plot_metric("avg_latency_ns",
            ylabel="Latency (ns per access)",
            title="Problem Size vs. Average Memory Latency",
            out_png="latency_vs_problem_size.png")

# EOF
