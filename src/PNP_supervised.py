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

# Boundary parameters for the inferencer
X_min, X_max = 0.0, 1.0
T_max = 0.2

@physicsnemo.sym.main(config_path="conf", config_name="config")
def run(cfg: PhysicsNeMoConfig) -> None:

    # 1. Network Architecture (Strictly identical to the PINN)
    net = FullyConnectedArch(
        input_keys=[Key("x"), Key("t")],
        output_keys=[Key("cp"), Key("cn"), Key("phi")],
        nr_layers=8,
        layer_size=256,
        activation_fn=torch.nn.Tanh(),
    )

    # We only need the network node. No PDE nodes since this is purely data-driven.
    nodes = [net.make_node(name="pnp_net")]

    # 2. Domain
    domain = Domain()

    # 3. Load the Benchmark Data (The FDM Ground Truth)
    # Ensure the path matches where you saved it in your explicit solver script
    data_path = "/content/drive/MyDrive/PINN/PNP/fipy_ground_truth.csv"
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"Could not find the ground truth data at {data_path}. Please check the path.")

    # 4. Prepare Data Dictionaries
    # Modulus requires dictionaries of numpy arrays shaped [N, 1]
    invar = {
        "x": df["x"].values.astype(np.float32).reshape(-1, 1),
        "t": df["t"].values.astype(np.float32).reshape(-1, 1)
    }

    outvar = {
        "cp": df["cp"].values.astype(np.float32).reshape(-1, 1),
        "cn": df["cn"].values.astype(np.float32).reshape(-1, 1),
        "phi": df["phi"].values.astype(np.float32).reshape(-1, 1)
    }

    # 5. Supervised Learning Constraint
    # This replaces all interior, boundary, and initial condition constraints
    data_constraint = PointwiseConstraint.from_numpy(
        nodes=nodes,
        invar=invar,
        outvar=outvar,
        batch_size=1024, # You can tweak this depending on your Colab GPU memory
    )
    domain.add_constraint(data_constraint, "supervised_data")

    # 6. Inferencer (Kept identical to the PINN for 1:1 evaluation)
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

    # 7. Solver
    solver = Solver(cfg, domain)
    solver.solve()

if __name__ == "__main__":
    run()
