from alpine.opt import bilevel
import jax.numpy as jnp
import numpy as np


def test_bilevel():
    target = np.array([2., 1.], dtype=np.float32)

    def statefun(x: np.array, y: np.array) -> float:
        """Discrete functional associated to the Minimum problem that determines the
        state.

        Args:
            x: state vector.
            y: paramters (controls).
        Returns:
            state vector that minimizes the functional.
        """
        return jnp.sum(jnp.square(jnp.square(x)-y))

    def objfun(x: np.array, y: np.array) -> float:
        """Objective function. Problem: choose y such that the state x(y) minimizes the
        distance wrt to the target.

        Args:
            x: state vector.
            y: paramters (controls).
        """
        return jnp.sum(jnp.square(x-target))

    # initial guesses
    x0 = np.ones(2)
    y0 = np.zeros(2)

    prb = bilevel.OptimizationProblem(objfun=objfun, state_en=statefun)
    x, y, fval = prb.run(x0, y0)
    print(x, y, fval)

    assert np.allclose(y, target**2)
    assert np.allclose(x, target)


if __name__ == "__main__":
    test_bilevel()
