import os
from dctkit.physics import burgers as b
import numpy as np
import dctkit
import alpine.data.util as u
from dctkit import config
import numpy.typing as npt
from typing import Dict, Tuple
import math
from dctkit.mesh import util
from dctkit.mesh.simplex import SimplicialComplex


data_path = os.path.dirname(os.path.realpath(__file__))

config()


def burgers_data(S: SimplicialComplex, t_max: float, dt: float,
                 u_0: npt.NDArray, nodes_BC: Dict, epsilon: float,
                 skip_dx: float, skip_dt: float,
                 scheme: str = "parabolic") -> Tuple[npt.NDArray, npt.NDArray]:
    # compute the solution u
    prb = b.Burgers(S, t_max, dt, u_0, nodes_BC, epsilon)
    prb.run(scheme)

    # extract data
    X = np.arange(prb.num_t_points/skip_dt, dtype=dctkit.int_dtype)
    y = prb.u_dot.T[::skip_dt, ::skip_dx]

    return X, y


if __name__ == "__main__":
    # SPACE PARAMS
    # spatial resolution
    dx = 2**4/2**8
    L = 2**4 + dx
    dx_norm = dx/L
    L_norm = 1
    #  Number of spatial grid points
    num_x_points = int(math.ceil(L / dx))
    num_x_points_norm = num_x_points

    # vector containing spatial points
    # x = np.linspace(0, L, num_x_points)
    x = np.linspace(-L/2, L/2, num_x_points)
    x_circ = (x[:-1] + x[1:])/2

    # initial velocity
    # u_0 = 2 * np.exp(-2 * (x_circ - 0.5 * L)**2)
    u_0 = 1 * np.exp(-1 * (x_circ + 0.5 * L/4)**2)
    umax = np.max(u_0)

    # TIME PARAMS
    T = 10
    T_norm = T*umax/L
    # temporal resolution
    dt = 10/2**9
    dt_norm = dt*umax/L
    # number of temporal grid points
    num_t_points_norm = int(math.ceil(T_norm / dt_norm))
    num_t_points = num_t_points_norm

    t = np.linspace(0, T, num_t_points)
    t_norm = np.linspace(0, T_norm, num_t_points_norm)

    # Viscosity
    # epsilon = 0.005*(L*umax)
    epsilon = 0.1
    epsilon_norm = epsilon/(L*umax)

    nodes_BC = {'left': np.zeros(num_t_points_norm),
                'right': np.zeros(num_t_points_norm)}

    skip_dx = 2**3
    skip_dt = 2**5

    # generate complex
    mesh, _ = util.generate_line_mesh(num_x_points_norm, L_norm)
    S = util.build_complex_from_mesh(mesh)
    S.get_hodge_star()
    S.get_flat_PDP_weights()

    u.save_dataset(data_generator=burgers_data,
                   data_generator_kwargs={'S': S,
                                          't_max': T_norm,
                                          'dt': dt_norm,
                                          'u_0': u_0/umax,
                                          'nodes_BC': nodes_BC,
                                          'epsilon': epsilon_norm,
                                          'skip_dx': skip_dx,
                                          'skip_dt': skip_dt,
                                          'scheme': "parabolic"},
                   perc_val=0.1,
                   perc_test=0.1,
                   format="npy",
                   shuffle=False)
