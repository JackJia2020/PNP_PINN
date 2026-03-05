import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

# ==========================================
# 1. Setup the Grid and Parameters
# ==========================================
N = 101                  # Number of spatial points
x = np.linspace(0, 1.0, N)
dx = x[1] - x[0]         # Distance between points

epsilon = 1            # Boundary layer thickness parameter
V_left, V_right = -0.5, 0.5

# We must use a microscopic time step for an "explicit" solver to not blow up
dt = 1e-5
T_max = 0.2
steps = int(T_max / dt)
save_every = int(0.01 / dt) # Save data every 0.01 seconds

# ==========================================
# 2. Initial Conditions
# ==========================================
# The Gaussian bump from your PINN
p = np.ones(N)
n = np.ones(N)
phi = np.linspace(V_left, V_right, N) # Straight line guess for potential

# Prepare the Poisson matrix ONCE
# This solves: epsilon * (phi[i-1] - 2*phi[i] + phi[i+1]) / dx^2 = -(p - n)
A = np.zeros((N, N))
A[0, 0] = 1.0            # Left boundary: 1 * phi[0] = V_left
A[-1, -1] = 1.0          # Right boundary: 1 * phi[-1] = V_right
for i in range(1, N - 1):
    A[i, i-1] = 1.0
    A[i, i]   = -2.0
    A[i, i+1] = 1.0

# Invert the matrix once to easily solve for potential later
A_inv = np.linalg.inv(A)

# ==========================================
# 3. The Time-Stepping Loop
# ==========================================
rows = []
print("Starting explicit simulation. This will take a moment but it is doing pure math...")

# timer
start_time = time.time()

for step in range(steps):

    # --- A. Solve the Poisson Equation for Potential (phi) ---
    # The right-hand side of the Poisson equation is the charge density: -(p - n)
    rhs = np.zeros(N)
    rhs[0] = V_left
    rhs[-1] = V_right
    # Multiply the interior by dx^2 / epsilon to isolate the phi terms
    rhs[1:-1] = -(p[1:-1] - n[1:-1]) * (dx**2) / epsilon

    # Calculate new potential
    phi = np.dot(A_inv, rhs)

    # --- B. Calculate Fluxes at the edges between nodes ---
    # J = -D * (dc/dx) - z * c * (dphi/dx)
    # We calculate this exactly halfway between node i and node i+1

    # Concentration gradients (dc/dx)
    dp_dx = (p[1:] - p[:-1]) / dx
    dn_dx = (n[1:] - n[:-1]) / dx
    dphi_dx = (phi[1:] - phi[:-1]) / dx

    # Average concentration at the edge
    p_edge = (p[1:] + p[:-1]) / 2.0
    n_edge = (n[1:] + n[:-1]) / 2.0

    # Total physical flux at each interior edge
    # z=+1 for p, z=-1 for n
    J_p_edge = -dp_dx - (+1) * p_edge * dphi_dx
    J_n_edge = -dn_dx - (-1) * n_edge * dphi_dx

    # --- C. Update Concentrations (Conservation of Mass) ---
    # Change in concentration = -(Flux out - Flux in) / dx
    # The boundaries are strictly 0 flux (blocking electrodes)

    J_p_full = np.concatenate(([0.0], J_p_edge, [0.0])) # Pad with 0 flux at walls
    J_n_full = np.concatenate(([0.0], J_n_edge, [0.0]))

    dp_dt = -(J_p_full[1:] - J_p_full[:-1]) / dx
    dn_dt = -(J_n_full[1:] - J_n_full[:-1]) / dx

    # Step forward in time: New = Old + (Rate of Change * dt)
    p = p + dp_dt * dt
    n = n + dn_dt * dt

    # --- D. Save Data ---
    if step % save_every == 0 or step == steps - 1:
        t_current = (step + 1) * dt
        for i in range(N):
            rows.append({"t": t_current, "x": x[i], "cp": p[i], "cn": n[i], "phi": phi[i]})

# timer stop
end_time = time.time()
print(f"Simulation finished in {end_time - start_time:.4f} seconds.")

# ==========================================
# 4. Save and Plot
# ==========================================
df = pd.DataFrame(rows)
df.to_csv("ground_truth.csv", index=False)
print("Saved ground_truth.csv")

plt.plot(x, p, label="Cations (cp)", color="blue")
plt.plot(x, n, label="Anions (cn)", color="red")
plt.title("Explicit Transparent Benchmark (t=0.2)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
