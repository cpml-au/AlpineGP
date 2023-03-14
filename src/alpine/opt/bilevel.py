import numpy as np
import jax.numpy as jnp
from jax import grad, jit, jacrev
from scipy.optimize import minimize


class BilevelOptimizationProblem():
    """Class for the bilevel optimization problem.

    Args:
        objfun: objective function to minimize wrt parameters (controls). Its
        arguments must be the state vector and the paramters vector.
        state_en: energy function to minimize wrt the state to define the current
        state given a set of controls.
    """

    def __init__(self, objfun: callable, state_en: callable) -> None:
        self.objfun = jit(objfun)
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
        """Wrapper for the objective function. Computes current state via state energy
        function minimization before calling the objective function.

        Args:
            a: array of paramters (controls).
        Returns:
            value of the objective function.
        """
        # computes current state as minimum energy state and coverts it to np.array to
        # be compatible with scipy minimization routines
        self.u = self.solve_state_equation(self.u, a).__array__()
        obj = self.objfun(self.u, a)
        return obj

    def solve_state_equation(self, u0: jnp.array, a0: jnp.array) -> jnp.array:
        # NOTE: for some reason jax.scipy.minimize is much slower than
        # scipy.optimize.minimize
        sol = minimize(self.state_en, u0, args=(a0,),
                       method="BFGS", jac=self.grad_u)
        return sol.x

    def solve_adjoint(self, u: jnp.array, y: jnp.array) -> jnp.array:
        """Solves the adjoint equation Ap=b, where A=(dF/du)^t, b=-dJ/da.

        Args:
            u: current state.
            a: parameters (controls).
        Returns:
            the adjoint state p.
        """
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
        """Runs bilevel optimization.

        Args:
            u0: initial guess for the state.
            a0: initial guess for the parameters (controls).
        """
        self.u = u0
        res = minimize(self.obj_fun_wrap, y0, method="BFGS",
                       jac=self.grad_obj_params)
        a = res.x
        u = self.solve_state_equation(u0, a).__array__()
        fval = 0.
        return u, a, fval
