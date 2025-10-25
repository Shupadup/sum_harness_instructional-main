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
# Compute MFLOPS ≈ (N-1 adds) / time / 1e6. Use small epsilon to avoid div-by-zero.
ops = [max(int(n) - 1, 1) for n in problem_sizes]
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

plt.show()

# EOF
