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


fname = "sample_data_3vars.csv"
df = pd.read_csv(fname, comment="#")
print(df)

var_names = list(df.columns)

print("var names =", var_names)

# split the df into individual vars
# assumption: column order - 0=problem size, 1=blas time, 2=basic time
# UPDATED: assume columns are [N, sum_direct, sum_indirect, sum_vector]

problem_sizes = df[var_names[0]].values.tolist()
code1_time = df[var_names[1]].values.tolist()  # sum_direct
code2_time = df[var_names[2]].values.tolist()  # sum_indirect
code3_time = df[var_names[3]].values.tolist()  # sum_vector

# ---------------- Runtime plot (seconds) ----------------
plt.title("Comparison of 3 Codes (Runtime)")
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

plt.xlabel("Problem Sizes (N)")
plt.ylabel("runtime (s)")

varNames = [var_names[1], var_names[2], var_names[3]]  # should be sum_direct, sum_indirect, sum_vector
plt.legend(varNames, loc="best")

plt.grid(axis='both')

# ---------------- MFLOPS plot (million adds per second) ----------------
# Compute MFLOPS ≈ (N adds) / time / 1e6. Use small epsilon to avoid div-by-zero.
ops = [max(int(n), 1) for n in problem_sizes]
EPS = 1e-12
code1_mflops = [(o / max(t, EPS)) / 1e6 for o, t in zip(ops, code1_time)]
code2_mflops = [(o / max(t, EPS)) / 1e6 for o, t in zip(ops, code2_time)]
code3_mflops = [(o / max(t, EPS)) / 1e6 for o, t in zip(ops, code3_time)]

plt.figure()
plt.title("Comparison of 3 Codes (MFLOPS)")
plt.xticks(xlocs, problem_sizes)
plt.plot(code1_mflops, "r-o")
plt.plot(code2_mflops, "b-x")
plt.plot(code3_mflops, "g-^")
#plt.yscale("log")

plt.xlabel("Problem Sizes (N)")
plt.ylabel("Throughput (MFLOPS)")
plt.legend(varNames, loc="best")
plt.grid(axis='both')

# ---------------- Additional metrics per problem size ----------------
# For each sum file:
#   - compute MFLOP/s from runtime and number of operations (done above)
#   - compute % of memory bandwidth utilized
#   - compute estimated memory latency in nanoseconds
#
# Assumptions for bytes moved per element:
#   sum_direct: the "direct" variant in class/slide sums loop index (no array load) -> 0 loads/elem.
#               If YOUR direct implementation reads A[i], set DIRECT_READS_PER_ELEM=1 below.
#   sum_indirect: 1 load/elem (pointer chasing A[idx])
#   sum_vector:   1 load/elem (sequential A[i])
#
# Set your platform peak memory bandwidth here (GB/s). Edit as needed.
#   - Laptop/desktop might be 30–80 GB/s.
#   - Perlmutter CPU nodes are on the order of ~200 GB/s per node (ballpark).
PEAK_BW_GBPS = 200.0

BYTES_PER_ELEM = 8  # int64_t
DIRECT_READS_PER_ELEM   = 0  # set to 1 if your sum_direct reads A[i]
INDIRECT_READS_PER_ELEM = 1
VECTOR_READS_PER_ELEM   = 1

def achieved_bw_gbps(bytes_moved, seconds):
    return (bytes_moved / max(seconds, EPS)) / 1e9

def bw_percent(achieved_gbps, peak_gbps):
    return 100.0 * achieved_gbps / max(peak_gbps, EPS)

def est_latency_ns(seconds, n_elems, reads_per_elem):
    if reads_per_elem <= 0 or n_elems <= 0:
        return float("nan")
    # time per load ~ total time / (#loads)
    loads = n_elems * reads_per_elem
    return (seconds / loads) * 1e9

# Compute bytes moved per N for each code
dir_bytes = [int(n) * BYTES_PER_ELEM * DIRECT_READS_PER_ELEM for n in problem_sizes]
ind_bytes = [int(n) * BYTES_PER_ELEM * INDIRECT_READS_PER_ELEM for n in problem_sizes]
vec_bytes = [int(n) * BYTES_PER_ELEM * VECTOR_READS_PER_ELEM for n in problem_sizes]

# Achieved bandwidth (GB/s)
dir_bw = [achieved_bw_gbps(b, t) for b, t in zip(dir_bytes, code1_time)]
ind_bw = [achieved_bw_gbps(b, t) for b, t in zip(ind_bytes, code2_time)]
vec_bw = [achieved_bw_gbps(b, t) for b, t in zip(vec_bytes, code3_time)]

# Percent of peak bandwidth
dir_bw_pct = [bw_percent(bw, PEAK_BW_GBPS) for bw in dir_bw]
ind_bw_pct = [bw_percent(bw, PEAK_BW_GBPS) for bw in ind_bw]
vec_bw_pct = [bw_percent(bw, PEAK_BW_GBPS) for bw in vec_bw]

# Estimated latency per memory access (ns)
dir_lat_ns = [est_latency_ns(t, int(n), DIRECT_READS_PER_ELEM) for n, t in zip(problem_sizes, code1_time)]
ind_lat_ns = [est_latency_ns(t, int(n), INDIRECT_READS_PER_ELEM) for n, t in zip(problem_sizes, code2_time)]
vec_lat_ns = [est_latency_ns(t, int(n), VECTOR_READS_PER_ELEM)   for n, t in zip(problem_sizes, code3_time)]

# Print a compact table of metrics for each problem size and code
print("\n==== Metrics per problem size ====")
hdr = (
    "N",
    "direct_time_s", "direct_MFLOPS", "direct_BW%","direct_lat_ns",
    "indirect_time_s", "indirect_MFLOPS", "indirect_BW%","indirect_lat_ns",
    "vector_time_s", "vector_MFLOPS", "vector_BW%", "vector_lat_ns",
)
print("{:>12s} | {:>13s} {:>14s} {:>11s} {:>14s} | {:>15s} {:>16s} {:>12s} {:>15s} | {:>13s} {:>14s} {:>12s} {:>14s}".format(*hdr))
for i, N in enumerate(problem_sizes):
    dlat = "NA" if pd.isna(dir_lat_ns[i]) else f"{dir_lat_ns[i]:.2f}"
    print("{:12d} | {:13.6f} {:14.2f} {:11.2f} {:14s} | {:15.6f} {:16.2f} {:12.2f} {:15.2f} | {:13.6f} {:14.2f} {:12.2f} {:14.2f}".format(
        int(N),
        code1_time[i], code1_mflops[i], dir_bw_pct[i], dlat,
        code2_time[i], code2_mflops[i], ind_bw_pct[i], ind_lat_ns[i],
        code3_time[i], code3_mflops[i], vec_bw_pct[i], vec_lat_ns[i]
    ))

# (Optional) also add plots for BW% and latency if you want visuals:
# Bandwidth %
plt.figure()
plt.title("Percent of Peak Memory Bandwidth Utilized")
plt.xticks(xlocs, problem_sizes)
plt.plot(dir_bw_pct, "r-o", label="sum_direct")
plt.plot(ind_bw_pct, "b-x", label="sum_indirect")
plt.plot(vec_bw_pct, "g-^", label="sum_vector")
plt.xlabel("Problem Sizes (N)")
plt.ylabel("% of Peak Bandwidth")
plt.grid(axis='both')
plt.legend(loc="best")

# Latency (ns per memory access)
plt.figure()
plt.title("Estimated Memory Latency (ns per load)")
plt.xticks(xlocs, problem_sizes)
plt.plot(dir_lat_ns, "r-o", label="sum_direct")
plt.plot(ind_lat_ns, "b-x", label="sum_indirect")
plt.plot(vec_lat_ns, "g-^", label="sum_vector")
plt.xlabel("Problem Sizes (N)")
plt.ylabel("Latency (ns)")
plt.grid(axis='both')
plt.legend(loc="best")

plt.tight_layout()
plt.show()

# EOF
