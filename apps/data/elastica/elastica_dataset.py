import os
import numpy as np
import dctkit as dt
import alpine.data.util as u

data_path = os.path.dirname(os.path.realpath(__file__))

# FIXME: FIX THE DOCS


def get_data_with_noise(noise):
    # get data
    abs_force = 5*np.arange(1, 11)
    y = -abs_force.astype(dt.float_dtype)
    X = np.empty((10, 10), dtype=dt.float_dtype)
    data_string = ["xy_F_" + str(i) + ".txt" for i in abs_force]
    for i, string in enumerate(data_string):
        data = np.loadtxt(os.path.join(data_path, string), dtype=float)
        # get data position
        x_data = data[:, 1][::10] + noise
        y_data = data[:, 2][::10] + noise
        # get theta
        theta_true = np.empty(10, dtype=dt.float_dtype)
        for j in range(10):
            theta_true[j] = np.arctan(
                (y_data[j+1]-y_data[j])/(x_data[j+1]-x_data[j]))
        # update X
        X[i, :] = theta_true
    return X, y


if __name__ == '__main__':
    # seet seed
    # FIXME: re-load xy_F_files, otherwise it doesn't work
    np.random.seed(42)
    u.save_dataset(data_generator=get_data_with_noise,
                   data_generator_kwargs={'noise': 0.005*np.random.rand(11)},
                   perc_val=0.3,
                   perc_test=0.2)
