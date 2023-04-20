import yaml
import os
import sys
from scipy import sparse
from dctkit.dec import cochain as C
from dctkit.mesh.simplex import SimplicialComplex
from dctkit.mesh.util import generate_1_D_mesh
import dctkit as dt
from alpine.gp import gpsymbreg as gps
from alpine.models.elastica import pset
from alpine.data.elastica_data import elastica_dataset as ed
import numpy as np
current = os.path.dirname(os.path.realpath(__file__))
parent_directory = os.path.dirname(current)

sys.path.append(parent_directory)


def elastica_img_from_string(config_file: dict, string: str, X_train, y_train, X_val, y_val):
    from stgp_elastica import plot_sol, eval_MSE, eval_fitness

    penalty = {"method": "length", "reg_param": 0.001}
    # get normalized simplicial complex
    S_1, x = generate_1_D_mesh(num_nodes=11, L=1.)
    S = SimplicialComplex(S_1, x, is_well_centered=True)
    S.get_circumcenters()
    S.get_primal_volumes()
    S.get_dual_volumes()
    S.get_hodge_star()

    # bidiagonal matrix to transform theta in (x,y)
    diag = [1]*(S.num_nodes)
    upper_diag = [-1]*(S.num_nodes-1)
    upper_diag[0] = 0
    diags = [diag, upper_diag]
    transform = sparse.diags(diags, [0, -1]).toarray()
    transform[1, 0] = -1

    # def h
    h = 1/(S.num_nodes-1)

    # get (x,y) coordinates for the dataset
    X = (X_train, X_val, X_test)
    x_all = []
    y_all = []
    for i in range(3):
        X_current = X[i]
        if X_current.ndim == 1:
            dim = X_current.ndim
            x_i = np.empty(S.num_nodes)
            y_i = np.empty(S.num_nodes)
        else:
            dim = X_current.shape[0]
            x_i = np.empty((dim, S.num_nodes))
            y_i = np.empty((dim, S.num_nodes))
        for j in range(dim):
            if dim == 1:
                theta_true = X_current
            else:
                theta_true = X_current[j, :]

            # reconstruct x_true and y_true
            cos_theta_true = h*np.cos(theta_true)
            sin_theta_true = h*np.sin(theta_true)
            b_x_true = np.insert(cos_theta_true, 0, 0)
            b_y_true = np.insert(sin_theta_true, 0, 0)
            if dim == 1:
                x_i = np.linalg.solve(transform, b_x_true)
                y_i = np.linalg.solve(transform, b_y_true)
            else:
                x_i[j, :] = np.linalg.solve(transform, b_x_true)
                y_i[j, :] = np.linalg.solve(transform, b_y_true)
        x_all.append(x_i)
        y_all.append(y_i)

    # get all theta_0
    theta_0_all = []
    for i in range(3):
        x_all_current = x_all[i]
        y_all_current = y_all[i]
        if x_all_current.ndim == 1:
            dim = x_all_current.ndim
            theta_0_current = np.empty(S.num_nodes-2, dtype=dt.float_dtype)
        else:
            dim = x_all_current.shape[0]
            theta_0_current = np.empty((dim, S.num_nodes-2), dtype=dt.float_dtype)
        for j in range(dim):
            if dim == 1:
                x_current = x_all_current
                y_current = y_all_current
            else:
                x_current = x_all_current[j, :]
                y_current = y_all_current[j, :]

            # def theta_0
            theta_0 = np.ones(S.num_nodes-2, dtype=dt.float_dtype)
            theta_0 *= np.arctan((y_current[-1] - y_current[1]) /
                                 (x_current[-1] - x_current[1]))
            if dim == 1:
                theta_0_current = theta_0
            else:
                theta_0_current[j, :] = theta_0
        theta_0_all.append(theta_0_current)

    # define internal cochain
    internal_vec = np.ones(S.num_nodes, dtype=dt.float_dtype)
    internal_vec[0] = 0.
    internal_vec[-1] = 0.
    internal_coch = C.CochainP0(complex=S, coeffs=internal_vec)

    # add it as a terminal
    pset.addTerminal(internal_coch, C.CochainP0, name="int_coch")

    # initialize toolbox and creator
    createIndividual, toolbox = gps.creator_toolbox_config(
        config_file=config_file, pset=pset)

    ind = createIndividual.from_string(string, pset)
    # estimate EI0
    EI0 = eval_MSE(ind, X_train, y_train, toolbox, S, theta_0_all[0], tune_EI0=True)
    ind.EI0 = EI0
    print(f"EI0: {EI0}")
    print(
        f"train_fit: {eval_fitness(ind, X_train, y_train, toolbox, S, theta_0_all[0], penalty)[0]}")
    print(
        f"MSE: {eval_MSE(ind, X_val, y_val, toolbox, S, theta_0_all[1], tune_EI0=False)}")
    plot_sol(ind, X_train, y_train, toolbox, S, theta_0_all[0], transform, False)
    plot_sol(ind, X_val, y_val, toolbox, S, theta_0_all[1], transform, False)
    plot_sol(ind, X_test, y_test, toolbox, S, theta_0_all[2], transform, False)


if __name__ == '__main__':
    n_args = len(sys.argv)
    assert n_args > 1, "Parameters filename needed."
    param_file = sys.argv[1]
    print("Parameters file: ", param_file)
    with open(param_file) as file:
        config_file = yaml.safe_load(file)
        print(yaml.dump(config_file))
    X_train, X_val, X_test, y_train, y_val, y_test = ed.load_dataset()
    # data_X, data_y = ed.get_data_with_noise(0.01*np.random.rand(11))
    # string = "Sub(MulF(1/2, InnP0(CochMulP0(int_coch, InvSt0(dD0(theta)), InvSt0(dD0(theta)))), InnD0(FL2_EI0, SinD0(theta))"
    # string = "InnD0(SinD0(theta), SquareD0(InvMulD0(SubD0(FL2_EI0, theta), SqrtF(InnP0(int_coch, InvSt0(ExpD1(SquareD1(dD0(theta)))))))))"
    # string = " InnD0(SinD0(theta), SquareD0(InvMulD0(SubD0(FL2_EI0, theta), InnP0(int_coch, InvSt0(ExpD1(SquareD1(dD0(theta))))))))"
    # string = "InnD0(SquareD0(CosD0(AddD0(FL2_EI0, theta))), theta)"
    # string = "Add(SinF(Add(Add(-1, 2), Add(CosF(InnD1(InvMulD1(CosD1(CochMulD1(SinD1(SqrtD1(St0(int_coch))), dD0(theta))), -1), InvMulD1(SubD1(dD0(FL2_EI0), St0(int_coch)), -1))), SqrtF(SinF(2))))), CosF(InnD0(MulD0(AddD0(theta, FL2_EI0), MulF(-1, -1)), ExpD0(theta))))"
    # string = " InnD0(SinD0(SinD0(SubD0(InvMulD0(FL2_EI0, 2), theta))), theta)"
    # string = "Add(SinF(SqrtF(Add(ExpF(CosF(InvF(SinF(2)))), Add(CosF(InnD1(CosD1(CochMulD1(SinD1(SqrtD1(St0(int_coch))), dD0(theta))), SubD1(dD0(FL2_EI0), St0(int_coch)))), SqrtF(SinF(2)))))), CosF(InnD0(AddD0(FL2_EI0, theta), ExpD0(InvMulD0(theta, 1/2)))))"
    # string = "Sub(InnP0(InvSt0(dD0(theta), InvSt0(dD0(theta))), InnD0(FL2_EI0, SinD0(theta))"
    # string = " SinF(CosF(InnD0(ExpD0(theta), FL2_EI0)))"
    # string = "InnD0(theta, SubD0(theta, theta))"
    string = "InnP0(ArccosP0(SubP0(SqrtP0(int_coch), int_coch)), SinP0(InvSt0(dD0(SqrtD0(CosD0(SubD0(FL2_EI0, theta)))))))"
    elastica_img_from_string(config_file, string=string,
                             X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val)
