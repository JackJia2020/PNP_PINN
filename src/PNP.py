import sys
import torch
import numpy as np
import physicsnemo.sym

from sympy import Symbol, Number, Function, exp, pi, Eq
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

# Define global symbolic variables for spatiotemporal coordinates
x = Symbol("x")
t = Symbol("t")

class PNP_1D(PDE):
    """
    1D Poisson-Nernst-Planck (PNP) PDE System.
    
    Models the drift-diffusion of cations (cp) and anions (cn) coupled with 
    the electrostatic potential (phi) via the Poisson equation.
    """
    name = "PNP_1D"

    def __init__(self, Dp=1.0, Dn=1.0, zp=1, zn=-1, epsilon=1):
        # Define dependent variables as functions of space and time
        cp = Function("cp")(x, t)
        cn = Function("cn")(x, t)
        phi = Function("phi")(x, t)

        # Convert physical parameters to SymPy numbers for symbolic differentiation
        Dp_sym = Number(Dp)
        Dn_sym = Number(Dn)
        zp_sym = Number(zp)
        zn_sym = Number(zn)
        eps_sym = Number(epsilon)

        self.equations = {}

        # 1. Flux Equations: Explicitly tracked to penalize non-physical boundary behaviors
        flux_p = -Dp_sym * cp.diff(x) - Dp_sym * zp_sym * cp * phi.diff(x)
        flux_n = -Dn_sym * cn.diff(x) - Dn_sym * zn_sym * cn * phi.diff(x)

        self.equations["flux_p"] = flux_p
        self.equations["flux_n"] = flux_n

        # 2. Nernst-Planck Equations: Conservation of mass for both ionic species
        self.equations["nernst_planck_p"] = (cp.diff(t) + flux_p.diff(x))
        self.equations["nernst_planck_n"] = (cn.diff(t) + flux_n.diff(x))

        # 3. Poisson Equation: Electrostatic potential driven by local charge density (rho)
        rho = zp_sym * cp + zn_sym * cn
        self.equations["poisson"] = (-eps_sym * phi.diff(x, x) - rho)

# --- Global Domain Parameters ---
X_min, X_max = 0.0, 1.0
T_max = 0.2
V_left, V_right = -0.5, 0.5 

@physicsnemo.sym.main(config_path="conf", config_name="config")
def run(cfg: PhysicsNeMoConfig) -> None:
    
    # ==========================================
    # 1. Architecture & PDE Setup
    # ==========================================
    
    # Initialize the surrogate neural network (Multi-Layer Perceptron)
    net = FullyConnectedArch(
        input_keys=[Key("x"), Key("t")],
        output_keys=[Key("cp"), Key("cn"), Key("phi")],
        nr_layers=8,
        layer_size=256,
        activation_fn=torch.nn.Tanh(), # Tanh ensures smooth, continuous derivatives
    )

    # Instantiate the PDE system and compile computational nodes
    pnp = PNP_1D(Dp=1.0, Dn=1.0, zp=1, zn=-1, epsilon=1)
    nodes = [net.make_node(name="pnp_net")] + pnp.make_nodes()

    # ==========================================
    # 2. Domain & Constraints Definition
    # ==========================================
    
    domain = Domain()
    geom = Line1D(X_min, X_max)

    # Initial Condition (IC): System starts at uniform concentration (cp=1, cn=1)
    ic_constraint = PointwiseInteriorConstraint(
        nodes=nodes,
        geometry=geom,
        outvar={"cp": Number(1.0), "cn": Number(1.0)},
        batch_size=1024,
        parameterization={t: 0.0}, 
        quasirandom=True,
        lambda_weighting={"cp": 100.0, "cn": 100.0} # High weight to strictly anchor t=0
    )
    domain.add_constraint(ic_constraint, "ic")

    # Global PDE Constraint: Enforce governing equations across the entire domain
    pde_constraint = PointwiseInteriorConstraint(
        nodes=nodes,
        geometry=geom,
        outvar={"nernst_planck_p": 0.0, "nernst_planck_n": 0.0, "poisson": 0.0},
        batch_size=2048,
        parameterization={t: (0.0, T_max)},
        quasirandom=True,
        lambda_weighting={"nernst_planck_p": 1.0, "nernst_planck_n": 1.0, "poisson": 1.0}
    )
    domain.add_constraint(pde_constraint, "pde_global")

    # High-Density Boundary Layers: Oversample the 5% margins to resolve steep gradients
    bl_width = 0.05
    geom_left_bl = Line1D(X_min, X_min + bl_width)
    geom_right_bl = Line1D(X_max - bl_width, X_max)

    for geom_bl, label in [(geom_left_bl, "left_bl"), (geom_right_bl, "right_bl")]:
        pde_bl_constraint = PointwiseInteriorConstraint(
            nodes=nodes,
            geometry=geom_bl,
            outvar={"nernst_planck_p": 0.0, "nernst_planck_n": 0.0, "poisson": 0.0},
            batch_size=1024,
            parameterization={t: (0.0, T_max)},
            quasirandom=True,
            lambda_weighting={"nernst_planck_p": 10.0, "nernst_planck_n": 10.0, "poisson": 10.0}
        )
        domain.add_constraint(pde_bl_constraint, f"pde_{label}")

    # Flux Boundary Conditions: Enforce zero flux (blocking electrodes) at x=0 and x=1
    for label, pos in [("left", X_min), ("right", X_max)]:
        bc_flux = PointwiseBoundaryConstraint(
            nodes=nodes,
            geometry=geom,
            outvar={"flux_p": 0.0, "flux_n": 0.0},
            batch_size=512,
            criteria=Eq(x, pos),
            parameterization={t: (0.0, T_max)},
            quasirandom=True,
            lambda_weighting={"flux_p": 50.0, "flux_n": 50.0}
        )
        domain.add_constraint(bc_flux, f"bc_flux_{label}")

    # Dirichlet Boundary Conditions: Enforce applied potential at the electrodes
    for label, pos, voltage in [("left", X_min, V_left), ("right", X_max, V_right)]:
        bc_phi = PointwiseBoundaryConstraint(
            nodes=nodes,
            geometry=geom,
            outvar={"phi": voltage},
            batch_size=256,
            criteria=Eq(x, pos),
            parameterization={t: (0.0, T_max)},
            quasirandom=True,
            lambda_weighting={"phi": 50.0}
        )
        domain.add_constraint(bc_phi, f"bc_phi_{label}")

    # ==========================================
    # 3. Evaluation & Execution
    # ==========================================
    
    # Inferencer: Evaluate the trained network on a dense uniform grid for post-processing
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

    # Initialize the solver with the specified configuration and execute training
    solver = Solver(cfg, domain)
    solver.solve()

if __name__ == "__main__":
    run()
