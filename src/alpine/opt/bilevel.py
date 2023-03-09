import numpy as np
import jax.numpy as jnp
from jax import grad, jit, jacrev
from jax.scipy.optimize import minimize
from scipy import optimize


class OptimizationProblem():
    def __init__(self, objfun: callable, state_en: callable) -> None:
        self.objfun = objfun
        self.state_en = jit(state_en)
        # gradient of the state energy wrt state
        self.grad_u = jit(grad(self.state_en))
        # array representing the lhs of the state equation
        self.F = self.grad_u
        # gradient of the state function wrt state
        self.dFdu = jit(jacrev(self.F))
        # gradient of the state function wrt parameters
        self.dFda = jit(jacrev(self.F, argnums=1))
        # gradient of the objective function wrt state
        self.dJdu = jit(grad(self.objfun))
        # current state
        self.u = None

    def obj_fun_wrap(self, a: np.array) -> float:
        self.u = self.solve_state_equation(self.u, a).__array__()
        obj = self.objfun(self.u, a)
        return obj

    def solve_state_equation(self, u0: jnp.array, y0: jnp.array) -> jnp.array:
        # TODO: use scipy
        sol = minimize(self.state_en, u0, args=(y0,), method="BFGS")
        return sol.x

    def solve_adjoint(self, u: jnp.array, y: jnp.array) -> jnp.array:
        """Solves the adjoint equation Ap=b, where A=(dF/du)^t, b=-dJ/da."""
        dFdu = self.dFdu(u, y)
        A = jnp.transpose(dFdu)
        b = -self.dJdu(u, y)
        p = jnp.linalg.solve(A, b)
        return p

    def grad_obj_params(self, a: jnp.array) -> np.array:
        """Computes the gradient of the objective function wrt the parameters using the
        solution of the adjoint equation.
        """
        p = self.solve_adjoint(self.u, a)
        dJda = jnp.matmul(jnp.transpose(self.dFda(self.u, a)), p)
        return dJda.__array__()

    def run(self, u0: np.array, y0: np.array) -> (np.array, np.array, float):
        """Runs bilevel optimization."""
        self.u = u0
        res = optimize.minimize(self.obj_fun_wrap, y0, method="BFGS",
                                jac=self.grad_obj_params)
        a = res.x
        u = self.solve_state_equation(u0, a).__array__()
        fval = 0.
        return u, a, fval
