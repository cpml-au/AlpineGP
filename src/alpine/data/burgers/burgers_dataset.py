import os
from dctkit.physics import burgers as b
import numpy as np
import dctkit
import alpine.data.util as u
from dctkit import config

data_path = os.path.dirname(os.path.realpath(__file__))

config()


def burgers_data(x_max, t_max, dx, dt, u_0, nodes_BC, epsilon, scheme="parabolic"):
    prb = b.Burgers(x_max, t_max, dx, dt, u_0, nodes_BC, epsilon)
    prb.run(scheme)
    X = np.arange(prb.num_t_points, dtype=dctkit.int_dtype)
    y = prb.u.T
    return X, y


if __name__ == "__main__":
    x_max = 5
    t_max = 2
    dx = 0.025
    dt = 0.001
    num_x_points = int(x_max/dx)
    num_t_points = int(t_max/dt)
    x = np.linspace(0, x_max, num_x_points)
    x_circ = (x[:-1] + x[1:])/2

    # boundary and initial conditions
    u_0 = 2 * np.exp(-2 * (x_circ - 0.5 * x_max)**2)
    nodes_BC = {'left': np.zeros(num_t_points), 'right': np.zeros(num_t_points)}

    # viscosity coefficient
    epsilon = 0.1
    u.save_dataset(data_generator=burgers_data,
                   data_generator_kwargs={'x_max': x_max, 't_max': t_max, 'dx': dx,
                                          'dt': dt, 'u_0': u_0, 'nodes_BC': nodes_BC,
                                          'epsilon': epsilon, 'scheme': "parabolic"},
                   perc_val=0.3,
                   perc_test=0.2,
                   format="npy")
