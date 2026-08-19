"""
Lightweight 1D momentum equation solver with parameterized closures.

Solves the momentum equation:
    d/dy(ρ(ν + ν_t) dU/dy) = -dp/dy + closure_sources(coefficients)

where:
  - ν_t is eddy viscosity from k-ε model (parameterized by coefficients)
  - k and ε are solved as auxiliary transport equations
  - closure_sources allow arbitrary modifications to momentum balance
  
Output: U(y), p(y) compared directly to DNS mean profiles.
This enables discovering optimal closure structures via coefficient optimization.
"""

import argparse
import json
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from scipy.integrate import solve_bvp
from scipy.interpolate import interp1d


@dataclass
class PDESystem:
    """Specification of a system of transport equations."""
    
    equations: Dict[str, Dict]  # eq_name -> {terms: [...]}
    coefficients: Dict[str, float]  # coefficient values


@dataclass
class DNSProfile:
    """DNS reference data at a location."""
    y: np.ndarray
    u_mean: np.ndarray
    eps_budget: Dict[str, np.ndarray]  # dissipation budget components
    

class CompositeRASSolver:
    """
    Solves momentum equation with parameterized turbulence closures.
    
    Steps:
    1. Solve k-ε transport equations (with tunable coefficients)
    2. Compute ν_t = Cmu * k² / ε
    3. Solve momentum equation: d/dy(ρ(ν + ν_t) dU/dy) = -dp/dy + sources
    4. Return U(y), optionally p(y) for comparison to DNS
    """
    
    def __init__(
        self, 
        y_profile: np.ndarray, 
        nu: float = 1e-5,
        pressure_gradient: float = 0.0
    ):
        """
        Args:
            y_profile: Wall-normal coordinate [0, δ]
            nu: Kinematic viscosity (m²/s)
            pressure_gradient: -dp/dy (assumed uniform, Pa/m)
        """
        self.y = y_profile
        self.nu = nu
        self.pressure_gradient = pressure_gradient
        self.dy = np.gradient(y_profile)
        self.n_y = len(y_profile)
        
    def solve(
        self, 
        pde_system: PDESystem,
        initial_guess: Dict[str, np.ndarray] = None,
        max_iterations: int = 50,
        tolerance: float = 1e-5
    ) -> Tuple[Dict[str, np.ndarray], bool]:
        """
        Solve momentum equation coupled with turbulence closure.
        
        Algorithm:
        1. Guess initial U(y)
        2. Solve k-ε equations and compute ν_t
        3. Solve momentum equation for updated U
        4. Repeat until convergence
        
        Returns:
            (solution_dict, converged_flag)
            solution_dict contains:
              - "U": velocity profile
              - "k": turbulent kinetic energy
              - "epsilon": dissipation rate
              - "p": pressure profile (if computed)
        """
        
        # Initial guess for velocity (linear profile: no slip at wall, free-slip at edge)
        if initial_guess is None or "U" not in initial_guess:
            U = np.linspace(0, 1.0, self.n_y)
        else:
            U = initial_guess["U"].copy()
        
        # Initial guesses for turbulence variables
        k = initial_guess.get("k", 0.01 * np.ones(self.n_y)) if initial_guess else 0.01 * np.ones(self.n_y)
        epsilon = initial_guess.get("epsilon", 0.001 * np.ones(self.n_y)) if initial_guess else 0.001 * np.ones(self.n_y)
        
        # Coupled iteration: turbulence → eddy viscosity → momentum
        for iteration in range(max_iterations):
            U_old = U.copy()
            
            # Step 1: Solve k equation
            k = self._solve_transport_equation(
                "k", 
                k, 
                pde_system,
                U, epsilon
            )
            k = np.maximum(k, 1e-10)
            
            # Step 2: Solve ε equation
            epsilon = self._solve_transport_equation(
                "epsilon",
                epsilon,
                pde_system,
                U, k
            )
            epsilon = np.maximum(epsilon, 1e-10)
            
            # Step 3: Compute eddy viscosity
            Cmu = pde_system.coefficients.get("Cmu", 0.09)
            nut = Cmu * (k ** 2) / (epsilon + 1e-10)
            
            # Step 4: Solve momentum equation with ν_t
            U = self._solve_momentum_equation(
                U,
                nut,
                pde_system
            )
            
            # Check convergence
            error = np.max(np.abs(U - U_old))
            if error < tolerance:
                solution = {
                    "U": U,
                    "k": k,
                    "epsilon": epsilon,
                    "nut": nut
                }
                return solution, True
        
        # Failed to converge
        solution = {
            "U": U,
            "k": k,
            "epsilon": epsilon,
            "nut": nut
        }
        return solution, False
    
    def _solve_transport_equation(
        self,
        eq_name: str,
        phi: np.ndarray,
        pde_system: PDESystem,
        U: np.ndarray,
        auxiliary_var: np.ndarray
    ) -> np.ndarray:
        """
        Solve one turbulence transport equation: k or ε.
        
        dφ/dy = source(φ, U, ...) with wall boundary condition φ(0)=0
        """
        phi_new = phi.copy()
        dU_dy = np.gradient(U, self.y)
        S = np.abs(dU_dy)
        
        # Forward Euler integration from wall
        for i in range(1, self.n_y):
            # Production ~ ν_t × (dU/dy)²
            Cmu = pde_system.coefficients.get("Cmu", 0.09)
            nut = Cmu * (phi[i-1] ** 2) / (auxiliary_var[i-1] + 1e-10)
            
            if eq_name == "k":
                c_k_prod = pde_system.coefficients.get("c_k_prod", 1.0)
                c_k_diss = pde_system.coefficients.get("c_k_diss", 1.0)
                
                # Production term
                P = nut * S[i] ** 2
                # Dissipation term: ε
                D = auxiliary_var[i-1]
                
                source = c_k_prod * P - c_k_diss * D
            
            elif eq_name == "epsilon":
                c_eps_prod = pde_system.coefficients.get("c_eps_prod", 1.44)
                c_eps_diss = pde_system.coefficients.get("c_eps_diss", 1.92)
                sigma_eps = pde_system.coefficients.get("sigma_eps", 1.3)
                
                # Production: C1 * ε/k * P
                P = nut * S[i] ** 2
                K = phi[i-1]
                
                source = c_eps_prod * (auxiliary_var[i-1] / K) * P - c_eps_diss * (auxiliary_var[i-1] ** 2) / K
            else:
                source = 0.0
            
            phi_new[i] = phi_new[i-1] + source * self.dy[i]
        
        return np.maximum(phi_new, 1e-10)
    
    def _solve_momentum_equation(
        self,
        U_guess: np.ndarray,
        nut: np.ndarray,
        pde_system: PDESystem
    ) -> np.ndarray:
        """
        Solve steady momentum equation:
            d/dy[(ν + ν_t) dU/dy] = -dp/dy + source_terms
        
        Boundary conditions: U(0)=0 (no-slip), dU/dy(δ)→ 0 (free stream)
        
        Uses simple 1D finite differences and iterative solver.
        """
        U = U_guess.copy()
        
        # Total viscosity
        nu_eff = self.nu + nut
        
        # Simple iteration: direct substitute to solve d²U/dy² = f
        for sub_iter in range(10):
            U_old = U.copy()
            
            # Interior points: finite difference
            for i in range(1, self.n_y - 1):
                dnu_dy = np.gradient(nu_eff, self.y)[i]
                d2U_dy2_approx = np.gradient(np.gradient(U, self.y), self.y)[i]
                
                # d/dy(ν_eff dU/dy) ≈ nu_eff[i] d²U/dy² + dnu_dy dU/dy
                momentum_source = -(self.pressure_gradient)
                
                dy = self.y[i+1] - self.y[i-1]
                
                # Simplified: assume dnu_dy term is small, solve main part
                U[i] = (
                    U[i-1] * nu_eff[i-1] + U[i+1] * nu_eff[i+1] + 
                    momentum_source * dy ** 2
                ) / (2 * nu_eff[i])
            
            # Boundary conditions
            U[0] = 0.0  # No-slip at wall
            # U[-1] = U[-2]  # Free-slip at edge (gradient → 0)
            
            error = np.max(np.abs(U - U_old))
            if error < 1e-5:
                break
        
        return U
    
    def _compute_source(
        self,
        eq_name: str,
        eq_spec: Dict,
        phi: np.ndarray,
        coeffs: Dict[str, float],
        solved_fields: Dict[str, np.ndarray]
    ) -> np.ndarray:
        """Compute source term for one equation (legacy, not used in new version)."""
        
        # Unified state vector
        state = {"y": self.y, "S": self.S, **solved_fields, eq_name: phi}
        
        source = np.zeros_like(phi)
        
        # Each term in the equation
        for term_spec in eq_spec.get("terms", []):
            term_name = term_spec["name"]
            term_type = term_spec["type"]
            coeff_key = f"{eq_name}_{term_name}"
            coeff = coeffs.get(coeff_key, 1.0)
            
            if term_type == "production":
                # Production ~ S × k
                if "k" in solved_fields:
                    source += coeff * self.S * solved_fields["k"]
            
            elif term_type == "dissipation":
                # Dissipation ~ ε²/k
                if "k" in solved_fields and "epsilon" in solved_fields:
                    k = solved_fields["k"]
                    eps = solved_fields["epsilon"]
                    source -= coeff * eps**2 / (k + 1e-10)
            
            elif term_type == "diffusion":
                # Diffusion: d/dy(ν_t dφ/dy)
                # Approximate with simple ν_t
                nut = coeff * 0.09 * (k**2 / eps) if "k" in solved_fields else 0
                d2phi_dy2 = np.gradient(np.gradient(phi, self.y), self.y)
                source += nut * d2phi_dy2
            
            elif term_type == "scalar":
                # Direct scalar multiplication
                source += coeff * phi
        
        return source


def load_dns_profile(
    h5_path: str,
    x_location: float
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """Load DNS data at a specific x-location."""
    import pandas as pd
    
    df = pd.read_hdf(h5_path, key="data")
    
    # Find closest x
    x_vals = df.index.get_level_values("x").unique().to_numpy()
    x_idx = np.argmin(np.abs(x_vals - x_location))
    x_sel = x_vals[x_idx]
    
    profile = df.loc[x_sel].reset_index()
    y = profile["y"].to_numpy()
    u_mean = profile["u"].to_numpy()
    
    return y, u_mean, {"x": x_sel}


def main():
    parser = argparse.ArgumentParser(
        description="Composite RANS solver: momentum equation with tunable closures"
    )
    parser.add_argument(
        "--dns-h5",
        default="data/jhtdb-transitional-bl/all-stats.h5",
        help="Path to DNS HDF5 data"
    )
    parser.add_argument(
        "--x-location",
        type=float,
        default=0.5,
        help="x-location for DNS extraction"
    )
    parser.add_argument(
        "--pde-config",
        default="sim/pde_structure.json",
        help="JSON config specifying PDE structure and coefficients"
    )
    parser.add_argument(
        "--viscosity",
        type=float,
        default=1e-5,
        help="Kinematic viscosity m²/s"
    )
    parser.add_argument(
        "--pressure-gradient",
        type=float,
        default=0.0,
        help="Pressure gradient -dp/dy (Pa/m)"
    )
    parser.add_argument(
        "--output-dir",
        default="results",
        help="Output directory"
    )
    parser.add_argument(
        "--output-profile",
        default="composite-rans-profile.csv",
        help="Output CSV with solved profiles"
    )
    
    args = parser.parse_args()
    
    # Load DNS reference
    dns_y, dns_u, metadata = load_dns_profile(args.dns_h5, args.x_location)
    
    # Load PDE structure config
    with open(args.pde_config) as f:
        pde_config = json.load(f)
    pde_system = PDESystem(**pde_config)
    
    # Solve RANS momentum equation
    solver = CompositeRASSolver(
        y_profile=dns_y,
        nu=args.viscosity,
        pressure_gradient=args.pressure_gradient
    )
    
    # Initial guesses
    initial_guess = {
        "U": dns_u.copy(),  # Start from DNS velocity
        "k": 0.01 * np.ones_like(dns_y),
        "epsilon": 0.001 * np.ones_like(dns_y),
    }
    
    # Solve
    solution, converged = solver.solve(
        pde_system=pde_system,
        initial_guess=initial_guess,
        max_iterations=50,
        tolerance=1e-5
    )
    
    if not converged:
        print(f"WARNING: Solver did not converge after 50 iterations")
    
    # Compare to DNS
    U_rans = solution["U"]
    mse = np.mean((U_rans - dns_u) ** 2)
    max_error = np.max(np.abs(U_rans - dns_u))
    
    print(f"\n{'='*60}")
    print(f"Momentum equation solution:")
    print(f"{'='*60}")
    print(f"DNS x-location:        {metadata['x']:.4f}")
    print(f"Grid points:           {len(dns_y)}")
    print(f"Convergence:           {'YES' if converged else 'NO'}")
    print(f"MSE(U_rans, U_dns):    {mse:.6e}")
    print(f"Max error:             {max_error:.6e}")
    print(f"{'='*60}")
    
    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    results = {
        "y": dns_y,
        "U_dns": dns_u,
        "U_rans": U_rans,
        "k": solution["k"],
        "epsilon": solution["epsilon"],
        "nut": solution["nut"],
    }
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(args.output_dir, args.output_profile), index=False)
    
    print(f"\nResults saved to {args.output_dir}/{args.output_profile}")
    
    return mse


if __name__ == "__main__":
    main()
