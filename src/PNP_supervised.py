import torch
import numpy as np
import pandas as pd

import physicsnemo.sym
from physicsnemo.sym.hydra import PhysicsNeMoConfig
from physicsnemo.sym.solver import Solver
from physicsnemo.sym.domain import Domain
from physicsnemo.sym.domain.constraint import PointwiseConstraint
from physicsnemo.sym.domain.inferencer import PointwiseInferencer
from physicsnemo.sym.models.fully_connected import FullyConnectedArch
from physicsnemo.sym.key import Key

# --- Global Domain Parameters ---
X_min, X_max = 0.0, 1.0
T_max = 0.2

@physicsnemo.sym.main(config_path="conf", config_name="config")
def run(cfg: PhysicsNeMoConfig) -> None:

    # ==========================================
    # 1. Architecture Setup
    # ==========================================
    
    # Initialize the surrogate neural network.
    # CRITICAL: This architecture (8 layers, 256 neurons, Tanh) is strictly 
    # identical to the PINN to ensure a fair 1:1 evaluation of the learning methods.
    net = FullyConnectedArch(
        input_keys=[Key("x"), Key("t")],
        output_keys=[Key("cp"), Key("cn"), Key("phi")],
        nr_layers=8,
        layer_size=256,
        activation_fn=torch.nn.Tanh(),
    )

    # Note: We only generate the network node here. Unlike the PINN, we do NOT 
    # instantiate any PDE nodes because this model is purely data-driven.
    nodes = [net.make_node(name="pnp_net")]

    domain = Domain()

    # ==========================================
    # 2. Data Loading & Train/Test Split
    # ==========================================
    
    
    # Load the FDM benchmark data using a relative path for GitHub portability.
    data_path = "./data/ground_truth.csv"
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"Could not find ground truth at {data_path}. Ensure you are running from the repo root.")

    # Withhold 20% of the data to prevent the model from purely memorizing the grid.
    # The network will only train on 80% of the FDM points, forcing it to generalize.
    df_train = df.sample(frac=0.8, random_state=42)

    # Format the training data into dictionaries of numpy arrays shaped [Batch, 1]
    # as required by the physicsnemo constraints.
    invar = {
        "x": df_train["x"].values.astype(np.float32).reshape(-1, 1),
        "t": df_train["t"].values.astype(np.float32).reshape(-1, 1)
    }

    outvar = {
        "cp": df_train["cp"].values.astype(np.float32).reshape(-1, 1),
        "cn": df_train["cn"].values.astype(np.float32).reshape(-1, 1),
        "phi": df_train["phi"].values.astype(np.float32).reshape(-1, 1)
    }

    # ==========================================
    # 3. Supervised Learning Constraint
    # ==========================================
    
    # This single constraint replaces all the complex PDE, Boundary, and Initial 
    # condition constraints used in the PINN. It directly minimizes the Mean Squared 
    # Error (MSE) between the network's predictions and the true FDM data.
    data_constraint = PointwiseConstraint.from_numpy(
        nodes=nodes,
        invar=invar,
        outvar=outvar,
        batch_size=1024,
    )
    domain.add_constraint(data_constraint, "supervised_data")

    # ==========================================
    # 4. Evaluation & Execution
    # ==========================================
    
    # Inferencer: Predict the solution over a dense, continuous 100x400 grid.
    # Because this grid contains 40,000 points, it evaluates the network on the 
    # 20% withheld FDM data PLUS thousands of entirely unseen spatial coordinates.
    inf_x = np.linspace(X_min, X_max, 100)
    inf_t = np.linspace(0, T_max, 400)
    mesh_x, mesh_t = np.meshgrid(inf_x, inf_t)
    
    invar_inf = {
        "x": mesh_x.flatten()[:, None].astype(np.float32),
        "t": mesh_t.flatten()[:, None].astype(np.float32),
    }
    
    inferencer = PointwiseInferencer(
        nodes=nodes,
        invar=invar_inf,
        output_names=["cp", "cn", "phi"],
        batch_size=1024,
    )
    domain.add_inferencer(inferencer, "inf_end")

    # Initialize the solver with the configuration and execute supervised training
    solver = Solver(cfg, domain)
    solver.solve()

if __name__ == "__main__":
    run()
