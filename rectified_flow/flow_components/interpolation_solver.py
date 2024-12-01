import torch
import torch.nn as nn
import sympy
from typing import Callable

from rectified_flow.utils import match_dim_with_data


class AffineInterpSolver:
    """
    Symbolic solver for the equations:
        x_t = a_t * x_1 + b_t * x_0
        dot_x_t = dot_a_t * x_1 + dot_b_t * x_0
    Given known variables and unknowns set as None, the solver computes the unknowns.
    """
    def __init__(self):
        # Define symbols
        x_0, x_1, x_t, dot_x_t = sympy.symbols('x_0 x_1 x_t dot_x_t')
        a_t, b_t, dot_a_t, dot_b_t = sympy.symbols('a_t b_t dot_a_t dot_b_t')
        
        # Equations
        eq1 = sympy.Eq(x_t, a_t * x_1 + b_t * x_0)
        eq2 = sympy.Eq(dot_x_t, dot_a_t * x_1 + dot_b_t * x_0)
        
        # Variables to solve for
        variables = [x_0, x_1, x_t, dot_x_t]
        self.symbolic_solvers = {}
        
        # Create symbolic solvers for all pairs of unknown variables
        for i in range(len(variables)):
            for j in range(i + 1, len(variables)):
                unknown1, unknown2 = variables[i], variables[j]
                # print(f"Solving for {unknown1} and {unknown2}")
                # Solve equations
                solutions = sympy.solve([eq1, eq2], (unknown1, unknown2), dict=True)
                if solutions:
                    solution = solutions[0]
                    # Create lambdified functions
                    expr1 = solution[unknown1]
                    expr2 = solution[unknown2]
                    func = sympy.lambdify(
                        [x_0, x_1, x_t, dot_x_t, a_t, b_t, dot_a_t, dot_b_t],
                        [expr1, expr2],
                        modules="numpy"
                    )
                    # Store solver function
                    var_names = (str(unknown1), str(unknown2))
                    self.symbolic_solvers[var_names] = func

    def solve(self, results):
        known_vars = {k: getattr(results, k) for k in ['x_0', 'x_1', 'x_t', 'dot_x_t'] if getattr(results, k) is not None}
        unknown_vars = {k: getattr(results, k) for k in ['x_0', 'x_1', 'x_t', 'dot_x_t'] if getattr(results, k) is None}
        unknown_keys = tuple(unknown_vars.keys())

        if len(unknown_keys) > 2:
            raise ValueError("At most two variables among (x_0, x_1, x_t, dot_x_t) can be unknown.")
        elif len(unknown_keys) == 0:
            return results
        elif len(unknown_keys) == 1:
            # Select one known variable to make up the pair
            for var in ['x_0', 'x_1', 'x_t', 'dot_x_t']:
                if var in known_vars:
                    unknown_keys.append(var)
                    break

        func = self.symbolic_solvers.get(unknown_keys)

        # Prepare arguments in the order [x_0, x_1, x_t, dot_x_t, a_t, b_t, dot_a_t, dot_b_t]
        args = []
        for var in ['x_0', 'x_1', 'x_t', 'dot_x_t', 'a_t', 'b_t', 'dot_a_t', 'dot_b_t']:
            value = getattr(results, var, None)
            if value is None:
                value = 0  # Placeholder for unknowns
            args.append(value)

        # Compute the unknown variables
        solved_values = func(*args)
        # Assign the solved values back to results
        setattr(results, unknown_keys[0], solved_values[0])
        setattr(results, unknown_keys[1], solved_values[1])

        return results


class AffineInterp(nn.Module):
    def __init__(
        self, 
        name: str = "straight",
        alpha: Callable | None = None,
        beta: Callable | None = None,
        dot_alpha: Callable | None = None,
        dot_beta: Callable | None = None,
    ):
        super().__init__()

        if name.lower() in ['straight', 'lerp']:
            # Special case for "straight" interpolation
            alpha = lambda t: t
            beta = lambda t: 1 - t
            dot_alpha = lambda t: torch.ones_like(t)
            dot_beta = lambda t: -torch.ones_like(t)
            name = 'straight'
        elif name.lower() in ['harmonic', 'cos', 'sin', 'slerp', 'spherical']:
            # Special case of "spherical" interpolation
            alpha = lambda t: torch.sin(t * torch.pi / 2.0)
            beta = lambda t: torch.cos(t * torch.pi / 2.0)
            dot_alpha = lambda t: torch.cos(t * torch.pi / 2.0) * torch.pi / 2.0
            dot_beta = lambda t: -torch.sin(t * torch.pi / 2.0) * torch.pi / 2.0
            name = 'spherical'
        elif name.lower() in ['ddim', 'ddpm']:
            # DDIM/DDPM scheme; see Eq 7 in https://arxiv.org/pdf/2209.03003
            a = 19.9
            b = 0.1
            alpha = lambda t: torch.exp(-a * (1 - t) ** 2 / 4.0 - b * (1 - t) / 2.0)
            beta = lambda t: torch.sqrt(1 - alpha(t) ** 2)
            name = 'DDIM'
        elif alpha is not None and beta is not None and dot_alpha is not None and dot_beta is not None:
            # Custom interpolation functions
            raise NotImplementedError("Custom interpolation functions are not yet supported.")

        self.name = name
        self.alpha = lambda t: alpha(self.ensure_tensor(t))
        self.beta = lambda t: beta(self.ensure_tensor(t))
        if dot_alpha is not None: self.dot_alpha = lambda t: dot_alpha(self.ensure_tensor(t))
        else: self.dot_alpha = None 
        if dot_beta is not None: self.dot_beta = lambda t: dot_beta(self.ensure_tensor(t))
        else: self.dot_beta = None 

        self.solver = AffineInterpSolver()
        self.a_t = None
        self.b_t = None
        self.dot_a_t = None
        self.dot_b_t = None
        self.x_0 = None
        self.x_1 = None
        self.x_t = None
        self.dot_x_t = None

    @staticmethod
    def ensure_tensor(x):
        return x if isinstance(x, torch.Tensor) else torch.tensor(x)

    @staticmethod
    def value_and_grad(f, input_tensor, detach=True):
        x = input_tensor.clone().detach().requires_grad_(True)
        with torch.enable_grad():
            value = f(x)
            grad, = torch.autograd.grad(value.sum(), x, create_graph=not detach)
        if detach:
            value = value.detach()
            grad = grad.detach()
        return value, grad

    def get_coeffs(self, t, detach=True):
        if self.dot_alpha is None:
            a_t, dot_a_t = self.value_and_grad(self.alpha, t, detach=detach)
        else:
            a_t = self.alpha(t)
            dot_a_t = self.dot_alpha(t)
        if self.dot_beta is None:
            b_t, dot_b_t = self.value_and_grad(self.beta, t, detach=detach)
        else:
            b_t = self.beta(t)
            dot_b_t = self.dot_beta(t)
        self.a_t = a_t
        self.b_t = b_t
        self.dot_a_t = dot_a_t
        self.dot_b_t = dot_b_t
        return a_t, b_t, dot_a_t, dot_b_t

    def forward(self, x_0, x_1, t, detach=True):
        t = match_dim_with_data(t, x_1.shape, device=x_1.device, dtype=x_1.dtype)
        a_t, b_t, dot_a_t, dot_b_t = self.get_coeffs(t, detach=detach)
        x_t = a_t * x_1 + b_t * x_0
        dot_x_t = dot_a_t * x_1 + dot_b_t * x_0
        return x_t, dot_x_t

    def solve(self, t=None, x_0=None, x_1=None, x_t=None, dot_x_t=None):
        """
        Solve equation: x_t = a_t*x_1+b_t*x_0, dot_x_t = dot_a_t*x_1+dot_b_t*x_0
        Set any of the known variables, and keep the unknowns as None; the solver will fill the unknowns.
        Example: interp.solve(t, x_t=x_t, dot_x_t=dot_x_t); print(interp.x_1.shape), print(interp.x_0.shape)
        """
        if t is None:
            raise ValueError("t must be provided")
        self.x_0 = x_0
        self.x_1 = x_1
        self.x_t = x_t
        self.dot_x_t = dot_x_t
        x_not_none = next((v for v in [x_0, x_1, x_t, dot_x_t] if v is not None), None)
        if x_not_none is None:
            raise ValueError("At least two of x_0, x_1, x_t, dot_x_t must not be None")
        t = match_dim_with_data(t, x_not_none.shape, device=x_not_none.device, dtype=x_not_none.dtype)
        a_t, b_t, dot_a_t, dot_b_t = self.get_coeffs(t)
        self.solver.solve(self)

        return self