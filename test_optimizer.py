# test_optimizer.py
import numpy as np
from src.optimizer import optimize_path

# Straight-line reference, big safe bubbles everywhere, generous d_max.
# The optimizer should return essentially q unchanged.
K = 10
q = np.column_stack([np.linspace(0, 9, K), np.zeros(K)])
bubbles = [(q[i, 0], q[i, 1], 0.5) for i in range(K)]

P_opt = optimize_path(q, bubbles, lambda_reg=1.0, d_max=1.5)

print("Max deviation from q:", np.max(np.linalg.norm(P_opt - q, axis=1)))
# Should be ~0 (like 1e-9).

# Zigzag reference that SHOULD get smoothed out.
q = np.array([[0, 0], [1, 1], [2, 0], [3, 1], [4, 0], [5, 1], [6, 0]], dtype=float)
bubbles = [(q[i, 0], q[i, 1], 0.8) for i in range(len(q))]

P_opt = optimize_path(q, bubbles, lambda_reg=1.0, d_max=2.0)
print("Original q:\n", q)
print("Optimized P:\n", np.round(P_opt, 3))