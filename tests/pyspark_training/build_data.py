import numpy as np

n = 10000
m = 7

data = np.random.randn(n, m) * np.array([0.5, 0.7, 20, -50, 5, 0.0001, -2])
col_names = [f"col_{chr(ord('A') + i)}" for i in range(7)]
np.savetxt(
    "data.csv", data, fmt="%.6f", delimiter=",", header=",".join(col_names), comments=""
)
