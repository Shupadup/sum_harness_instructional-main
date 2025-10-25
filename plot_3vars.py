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
import matplotlib.pyplot as plt


fname = "perlmutter_all_methods_metrics.csv"
df = pd.read_csv(fname, comment="#")
print(df)

var_names = list(df.columns)

print("var names =", var_names)

# split the df into individual vars
# assumption: column order - 0=problem size, 1=blas time, 2=basic time
# UPDATED: assume columns are [N, sum_direct, sum_indirect, sum_vector] and values are *runtime in seconds*
problem_sizes = df[var_names[0]].values.tolist()
code1_time = df[var_names[1]].values.tolist()  # sum_direct runtime (s)
code2_time = df[var_names[2]].values.tolist()  # sum_indirect runtime (s)
code3_time = df[var_names[3]].values.tolist()  # sum_vector runtime (s)

# ---------------- Existing runtime plot (kept as-is) ----------------
plt.title("Comparison of 3 Codes")
xlocs = [i for i in range(len(problem_sizes))]
plt.xticks(xlocs, problem_sizes)

# here, we are plotting the raw values read from the input .csv file, which
# we interpret as being "time" that maps directly to the y-axis.
#
# what if we want to plot MFLOPS instead? How do we compute MFLOPS from
# time and problem size? You may need to add some code here to compute
# MFLOPS, then modify the plt.plot() lines below to plot MFLOPS rather than time.

plt.plot(code1_time, "r-o")
plt.plot(code2_time, "b-x")
plt.plot(code3_time, "g-^")

#plt.xscale("log")
#plt.yscale("log")

plt.xlabel("Problem Sizes")
plt.ylabel("runtime")

varNames = [var_names[1], var_names[2], var_names[3]]
plt.legend(varNames, loc="best")
plt.grid(axis='both')
plt.show()

# ---------------- New: compute MFLOP/s, % peak bandwidth, and memory latency ----------------
# Assumptions per assignment: loop index & accumulator are "free" (in regs), count only loop body work.
# FLOPs per iteration: 1 addition for all three.
# Memory accesses per iteration:
#   - sum_direct:   0 (s += i)
#   - sum_indirect: 2 (s += A[idx]; idx = A[idx])
#   - sum_vector:   1 (s += A[i])
BYTES_PER_ELEM = 8  # int64_t
PEAK_BW_GBPS = 409.6  # Perlmutter CPU node theoretical peak (2 sockets × 204.8 GB/s)
EPS = 1e-12

# Convert to arrays for convenience
import numpy as np
N = np.asarray(problem_sizes, dtype=np.int64)
t_direct   = np.asarray(code1_time, dtype=float)
t_indirect = np.asarray(code2_time, dtype=float)
t_vector   = np.asarray(code3_time, dtype=float)

# MFLOP/s = (ops / time) where ops = N adds, reported in *millions* of ops per second
direct_mflops   = np.where(t_direct   > 0, (N / t_direct)   / 1e6, np.nan)
indirect_mflops = np.where(t_indirect > 0, (N / t_indirect) / 1e6, np.nan)
vector_mflops   = np.where(t_vector   > 0, (N / t_vector)   / 1e6, np.nan)

# Bytes moved in the loop
bytes_direct   = N * 0 * BYTES_PER_ELEM
bytes_indirect = N * 2 * BYTES_PER_ELEM
bytes_vector   = N * 1 * BYTES_PER_ELEM

# Achieved bandwidth (GB/s) and % of peak
bw_direct_gbps   = np.where(t_direct   > 0, (bytes_direct   / t_direct)   / 1e9, 0.0)
bw_indirect_gbps = np.where(t_indirect > 0, (bytes_indirect / t_indirect) / 1e9, 0.0)
bw_vector_gbps   = np.where(t_vector   > 0, (bytes_vector   / t_vector)   / 1e9, 0.0)

bw_direct_pct   = np.where(PEAK_BW_GBPS > 0, 100.0 * bw_direct_gbps   / PEAK_BW_GBPS, 0.0)
bw_indirect_pct = np.where(PEAK_BW_GBPS > 0, 100.0 * bw_indirect_gbps / PEAK_BW_GBPS, 0.0)
bw_vector_pct   = np.where(PEAK_BW_GBPS > 0, 100.0 * bw_vector_gbps   / PEAK_BW_GBPS, 0.0)

# Average memory latency (ns/access) = time / accesses
acc_direct   = N * 0
acc_indirect = N * 2
acc_vector   = N * 1

lat_direct_ns   = np.where(acc_direct   > 0, (t_direct   / acc_direct)   * 1e9, np.nan)
lat_indirect_ns = np.where(acc_indirect > 0, (t_indirect / acc_indirect) * 1e9, np.nan)
lat_vector_ns   = np.where(acc_vector   > 0, (t_vector   / acc_vector)   * 1e9, np.nan)

# ---------------- Plot 1: MFLOP/s ----------------
plt.figure()
plt.title("Problem Size vs. MFLOP/s")
xlocs = [i for i in range(len(N))]
plt.xticks(xlocs, N)
plt.plot(direct_mflops,   "r-o", label=var_names[1])  # sum_direct
plt.plot(indirect_mflops, "b-x", label=var_names[2])  # sum_indirect
plt.plot(vector_mflops,   "g-^", label=var_names[3])  # sum_vector
#plt.xscale("log"); plt.yscale("log")
plt.xlabel("Problem Size (N)")
plt.ylabel("MFLOP/s")
plt.grid(True, which="both")
plt.legend(loc="best")
# plt.savefig("mflops_vs_problem_size.png", dpi=200)

# ---------------- Plot 2: % Peak Memory Bandwidth ----------------
plt.figure()
plt.title("Problem Size vs. % Peak Memory Bandwidth Utilized")
plt.xticks(xlocs, N)
plt.plot(bw_direct_pct,   "r-o", label=var_names[1])  # sum_direct
plt.plot(bw_indirect_pct, "b-x", label=var_names[2])  # sum_indirect
plt.plot(bw_vector_pct,   "g-^", label=var_names[3])  # sum_vector
plt.xlabel("Problem Size (N)")
plt.ylabel("% of Peak Memory Bandwidth")
plt.grid(True, which="both")
plt.legend(loc="best")
# plt.savefig("bandwidth_pct_vs_problem_size.png", dpi=200)

# ---------------- Plot 3: Memory Latency (ns/access) ----------------
plt.figure()
plt.title("Problem Size vs. Average Memory Latency")
plt.xticks(xlocs, N)
plt.plot(lat_direct_ns,   "r-o", label=var_names[1])  # sum_direct
plt.plot(lat_indirect_ns, "b-x", label=var_names[2])  # sum_indirect
plt.plot(lat_vector_ns,   "g-^", label=var_names[3])  # sum_vector
plt.xlabel("Problem Size (N)")
plt.ylabel("Latency (ns per access)")
plt.grid(True, which="both")
plt.legend(loc="best")
# plt.savefig("latency_vs_problem_size.png", dpi=200)

plt.tight_layout()
plt.show()

# EOF
