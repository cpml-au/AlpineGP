import os
from dctkit.physics import burgers as b
import numpy as np
import dctkit
import alpine.data.util as u
from dctkit import config
import numpy.typing as npt
from typing import Dict, Tuple
import math

data_path = os.path.dirname(os.path.realpath(__file__))

config()


def burgers_data(x_max: float, t_max: float, dx: float, dt: float,
                 u_0: npt.NDArray, nodes_BC: Dict, epsilon: float,
                 skip_dx: float, skip_dt: float,
                 scheme: str = "parabolic") -> Tuple[npt.NDArray, npt.NDArray]:
    prb = b.Burgers(x_max, t_max, dx, dt, u_0, nodes_BC, epsilon)
    prb.run(scheme)
    X = np.arange(prb.num_t_points/skip_dt, dtype=dctkit.int_dtype)
    y = prb.u.T[::skip_dt, ::skip_dx]
    return X, y


if __name__ == "__main__":
    # SPACE PARAMS
    # spatial resolution
    dx = 5/2**8
    L = 5 + dx
    dx_norm = dx/L
    L_norm = 1
    #  Number of spatial grid points
    num_x_points = int(math.ceil(L / dx))
    num_x_points_norm = num_x_points

    # vector containing spatial points
    x = np.linspace(0, L, num_x_points)
    x_circ = (x[:-1] + x[1:])/2

    # initial velocity
    u_0 = 2 * np.exp(-2 * (x_circ - 0.5 * L)**2)
    umax = np.max(u_0)

    # TIME PARAMS
    T = 2
    T_norm = T*umax/L
    # temporal resolution
    dt = 2/2**10
    dt_norm = dt*umax/L
    # number of temporal grid points
    num_t_points_norm = int(math.ceil(T_norm / dt_norm))

    # Viscosity
    epsilon = 0.05
    epsilon_norm = epsilon/(L*umax)

    # define skip_dx and skip_dt
    skip_dx = 2**2
    skip_dt = 2**5

    nodes_BC = {'left': np.zeros(num_t_points_norm),
                'right': np.zeros(num_t_points_norm)}

    u.save_dataset(data_generator=burgers_data,
                   data_generator_kwargs={'x_max': L_norm,
                                          't_max': T_norm,
                                          'dx': dx_norm,
                                          'dt': dt_norm,
                                          'u_0': u_0/umax,
                                          'nodes_BC': nodes_BC,
                                          'epsilon': epsilon_norm,
                                          'skip_dx': skip_dx,
                                          'skip_dt': skip_dt,
                                          'scheme': "parabolic"},
                   perc_val=0.3,
                   perc_test=0.2,
                   format="npy")
