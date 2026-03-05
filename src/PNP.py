from sympy import Symbol, Number, Function, exp, pi
import torch
import numpy as np

import physicsnemo.sym
from physicsnemo.sym.hydra import PhysicsNeMoConfig
from physicsnemo.sym.solver import Solver
from physicsnemo.sym.domain import Domain
from physicsnemo.sym.geometry.primitives_1d import Line1D
from physicsnemo.sym.domain.constraint import (
    PointwiseBoundaryConstraint,
    PointwiseInteriorConstraint,
)
from physicsnemo.sym.domain.inferencer import PointwiseInferencer
from physicsnemo.sym.models.fully_connected import FullyConnectedArch
from physicsnemo.sym.key import Key
from physicsnemo.sym.eq.pde import PDE
from physicsnemo.sym.graph import Graph
from sympy import Eq
from physicsnemo.sym.models.modified_fourier_net import ModifiedFourierNetArch
from physicsnemo.sym.node import Node

# Define symbolic variables
x = Symbol("x")
t = Symbol("t")

class PNP_1D(PDE):
    """
    Poisson-Nernst-Planck System (1D)
    """
    name = "PNP_1D"

    def __init__(self, Dp=1.0, Dn=1.0, zp=1, zn=-1, epsilon=1):
        # Dependent variables: Cation (cp), Anion (cn), Potential (phi)
        cp = Function("cp")(x, t)
        cn = Function("cn")(x, t)
        phi = Function("phi")(x, t)

        # Parameters
        Dp_sym = Number(Dp)
        Dn_sym = Number(Dn)
        zp_sym = Number(zp)
        zn_sym = Number(zn)
        eps_sym = Number(epsilon)

        self.equations = {}

        # 1. Calculate the Total Flux explicitly
        flux_p = -Dp_sym * cp.diff(x) - Dp_sym * zp_sym * cp * phi.diff(x)
        flux_n = -Dn_sym * cn.diff(x) - Dn_sym * zn_sym * cn * phi.diff(x)

        # 2. Add the Flux to the equations dictionary so the network outputs it
        self.equations["flux_p"] = flux_p
        self.equations["flux_n"] = flux_n

        # 3. Nernst-Planck Equations
        self.equations["nernst_planck_p"] = (cp.diff(t) + flux_p.diff(x))
        self.equations["nernst_planck_n"] = (cn.diff(t) + flux_n.diff(x))

        # 4. Poisson Equation
        rho = zp_sym * cp + zn_sym * cn
        self.equations["poisson"] = (-eps_sym * phi.diff(x, x) - rho)

# --- CRITICAL FIX: Align with FDM Benchmark Parameters ---
X_min, X_max = 0.0, 1.0
T_max = 0.2
V_left, V_right = -0.5, 0.5 # Matches your explicit solver exactly

@physicsnemo.sym.main(config_path="conf", config_name="config")
def run(cfg: PhysicsNeMoConfig) -> None:

    # 1. Network Architecture
    net = FullyConnectedArch(
        input_keys=[Key("x"), Key("t")],
        output_keys=[Key("cp"), Key("cn"), Key("phi")],
        nr_layers=8,
        layer_size=256,
        activation_fn=torch.nn.Tanh(),
    )

    # 2. Define PDE
    pnp = PNP_1D(Dp=1.0, Dn=1.0, zp=1, zn=-1, epsilon=1)

    # Sympy expression for the initial condition
    initial_c = Number(1.0)

    # Nodes array now only contains the network and PDE (ansatz nodes removed)
    nodes = [net.make_node(name="pnp_net")] + pnp.make_nodes()

    # 3. Domain & Geometry
    domain = Domain()
    geom = Line1D(X_min, X_max)

    # 4. Initial Condition (Soft Constraint)
    ic_constraint = PointwiseInteriorConstraint(
        nodes=nodes,
        geometry=geom,
        outvar={"cp": initial_c, "cn": initial_c},
        batch_size=1024,
        parameterization={t: 0.0}, # Fix time at exactly t=0
        quasirandom=True,
        lambda_weighting={
            "cp": 100.0,
            "cn": 100.0
        }
    )
    domain.add_constraint(ic_constraint, "ic")

    # 5. Interior Constraints (The PDEs)
    pde_constraint = PointwiseInteriorConstraint(
        nodes=nodes,
        geometry=geom,
        outvar={
            "nernst_planck_p": 0.0,
            "nernst_planck_n": 0.0,
            "poisson": 0.0
        },
        batch_size=2048,
        parameterization={t: (0.0, T_max)},
        quasirandom=True,
        lambda_weighting={
            "nernst_planck_p": 1.0,
            "nernst_planck_n": 1.0,
            "poisson": 1.0
        }
    )
    domain.add_constraint(pde_constraint, "pde")

    # --- High-Density Boundary Layer Constraints ---
    # Define the thickness of the boundary layer we want to focus on (e.g., 5% of the domain)
    bl_width = 0.05

    # Create sub-geometries specifically for the regions near the electrodes
    geom_left_bl = Line1D(X_min, X_min + bl_width)
    geom_right_bl = Line1D(X_max - bl_width, X_max)

    # 5a. Left Boundary Layer Interior PDE
    pde_constraint_left_bl = PointwiseInteriorConstraint(
        nodes=nodes,
        geometry=geom_left_bl,
        outvar={
            "nernst_planck_p": 0.0,
            "nernst_planck_n": 0.0,
            "poisson": 0.0
        },
        batch_size=1024, # High density: 1024 points packed into just 5% of the space
        parameterization={t: (0.0, T_max)},
        quasirandom=True,
        # Optional: Force the optimizer to care more about these points
        lambda_weighting={
            "nernst_planck_p": 10.0,
            "nernst_planck_n": 10.0,
            "poisson": 10.0
        }
    )
    domain.add_constraint(pde_constraint_left_bl, "pde_left_bl")

    # 5b. Right Boundary Layer Interior PDE
    pde_constraint_right_bl = PointwiseInteriorConstraint(
        nodes=nodes,
        geometry=geom_right_bl,
        outvar={
            "nernst_planck_p": 0.0,
            "nernst_planck_n": 0.0,
            "poisson": 0.0
        },
        batch_size=1024,
        parameterization={t: (0.0, T_max)},
        quasirandom=True,
        lambda_weighting={
            "nernst_planck_p": 10.0,
            "nernst_planck_n": 10.0,
            "poisson": 10.0
        }
    )
    domain.add_constraint(pde_constraint_right_bl, "pde_right_bl")

    # 6. Boundary Conditions
    # Flux BCs (Blocking Electrodes)
    for label, pos in [("left", X_min), ("right", X_max)]:
        bc_flux = PointwiseBoundaryConstraint(
            nodes=nodes,
            geometry=geom,
            outvar={"flux_p": 0.0, "flux_n": 0.0},
            batch_size=512,
            criteria=Eq(x, pos),
            parameterization={t: (0.0, T_max)},
            quasirandom=True,
            lambda_weighting={
                "flux_p": 50.0,
                "flux_n": 50.0
            }
        )
        domain.add_constraint(bc_flux, f"bc_flux_{label}")

    # Proper Dirichlet BCs for Potential
    bc_phi_left = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=geom,
        outvar={"phi": V_left},
        batch_size=256,
        criteria=Eq(x, X_min),
        parameterization={t: (0.0, T_max)},
        quasirandom=True,
        lambda_weighting={
            "phi": 50.0
        }
    )
    domain.add_constraint(bc_phi_left, "bc_phi_left")

    bc_phi_right = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=geom,
        outvar={"phi": V_right},
        batch_size=256,
        criteria=Eq(x, X_max),
        parameterization={t: (0.0, T_max)},
        quasirandom=True,
        lambda_weighting={
            "phi": 50.0
        }
    )
    domain.add_constraint(bc_phi_right, "bc_phi_right")

    # 7. Inferencer
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
        batch_size=4096,
    )
    domain.add_inferencer(inferencer, "inf_end")

    # 8. Solver
    solver = Solver(cfg, domain)
    solver.solve()

if __name__ == "__main__":
    run()
