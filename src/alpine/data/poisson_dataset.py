import numpy as np
# import multiprocessing
from deap import tools


def generate_dataset(S, mult):
    """Generate a dataset for the Poisson problem.

    Args:
        S (SimplicialComplex): simplicial complex where the functions of the dataset
        are defined.
        mult (int): the multiplicity of every class (for now 3) of functions of the
        dataset.

    Returns:
        (np.array): np.array of the dataset samples.
        (np.array): np.array of the labels.
    """
    node_coords = S.node_coord
    num_nodes = S.num_nodes
    data_X = np.empty((3*mult, num_nodes))
    data_y = np.empty((3*mult, num_nodes))
    for i in range(mult):
        # ith quadratic function
        q_i = 1/(i + 1)**2 * (node_coords[:, 0]**2 + node_coords[:, 1]**2)
        rhs_qi = (4/(i+1)**2) * np.ones(num_nodes)

        # ith exponential function
        trig_i = np.cos(i*node_coords[:, 0]) + np.sin(i*node_coords[:, 1])
        rhs_trigi = -i**2 * trig_i

        # ith power function
        p_i = node_coords[:, 0]**(i+2) + node_coords[:, 1]**(i+2)
        rhs_pi = (i+2)*(i+1)*(node_coords[:, 0]**(i) + node_coords[:, 1]**(i))

        # fill the dataset
        data_X[3*i, :] = q_i
        data_X[3*i+1, :] = trig_i
        data_X[3*i+2, :] = p_i
        data_y[3*i, :] = rhs_qi
        data_y[3*i+1, :] = rhs_trigi
        data_y[3*i+2, :] = rhs_pi

    return data_X, data_y


def poisson_model_selection(GPproblem, evalPoisson, X_train, y_train, kf):
    # start learning
    # pool = multiprocessing.Pool()
    # GPproblem.toolbox.register("map", pool.map)
    best_individuals = []
    best_train_scores = []
    best_val_scores = []
    for train_index, valid_index in kf.split(X_train, y_train):
        # divide the dataset in training and validation set
        X_t, X_val = X_train[train_index, :], X_train[valid_index, :]
        y_t, y_val = y_train[train_index, :], y_train[valid_index, :]

        GPproblem.toolbox.register("evaluate", evalPoisson, X=X_t, y=y_t)

        # train the model in the training set
        # pool = multiprocessing.Pool()
        # GPproblem.toolbox.register("map", pool.map)
        GPproblem.run(plot_history=True,
                      print_log=True,
                      plot_best=True,
                      seed=None)
        # pool.close()

        # Print best individual
        best = tools.selBest(GPproblem.pop, k=1)
        print(f"The best individual in this fold is {str(best[0])}")

        # evaluate score on the current training and validation set
        score_train = GPproblem.min_history[-1]
        score_val = evalPoisson(best[0], X_val, y_val)

        print(f"The best score on training set in this fold is {score_train}")
        print(f"The best score on validation set in this fold is {score_val}")

        # save best individual and best score on training and validation set
        best_individuals.append(best[0])

        # FIXME: do I need it?
        best_train_scores.append(score_train)
        best_val_scores.append(score_train)

        print("-FOLD COMPLETED-")

    return best_individuals
