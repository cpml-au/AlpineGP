import matplotlib.pyplot as plt
from deap import gp
from dctkit import config
from dctkit.math.opt import optctrl as oc
from alpine.gp import gpsymbreg as gps
from alpine.data import Dataset
from alpine.gp import util
import numpy as np
import ray
import yaml
from typing import Tuple, Callable, List
import numpy.typing as npt
import jax.numpy as jnp
import time
from jax import jit, grad
import warnings
import pygmo as pg
from functools import partial
import re
from pmlb import fetch_data
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

num_cpus = 1
num_runs = 1  # 20

# config()


def check_trig_fn(ind):
    return len(re.findall("cos", str(ind))) + len(re.findall("sin", str(ind)))


def check_nested_trig_fn(
    ind): return util.detect_nested_trigonometric_functions(str(ind))


def eval_model(individual, Xs, dataset, num_variables, consts=[]):
    warnings.filterwarnings("ignore")
    if num_variables == 1:
        y_pred = individual(dataset.X, consts)
    elif num_variables == 2:
        y_pred = individual(dataset.X[:, 0], dataset.X[:, 1], consts)
    else:
        y_pred = individual(*Xs, consts)

    return y_pred


def compute_MSE(individual, Xs, D, num_variables, consts=[]):
    y_pred = eval_model(individual, Xs, D, num_variables, consts)
    MSE = np.sum((D.y-y_pred)**2)

    if np.isnan(MSE) or np.isinf(MSE):
        MSE = 1e8

    MSE *= 1/D.X.shape[0]
    return MSE


# TODO: this could become a library function


def compile_individual_with_consts(tree, toolbox, special_term_name="a"):
    const_idx = 0
    tree_clone = toolbox.clone(tree)
    for i, node in enumerate(tree_clone):
        if isinstance(node, gp.Terminal) and node.name[0:3] != "ARG":
            if node.name == special_term_name:
                new_node_name = special_term_name + "[" + str(const_idx) + "]"
                tree_clone[i] = gp.Terminal(new_node_name, True, float)
                const_idx += 1

    individual = toolbox.compile(
        expr=tree_clone, extra_args=[special_term_name])
    return individual, const_idx


def eval_MSE_and_tune_constants(tree, toolbox, D, num_variables):
    individual, num_consts = compile_individual_with_consts(tree, toolbox)
    if num_variables > 1:
        Xs = [D.X[:, i] for i in range(num_variables)]
    else:
        Xs = [D.X]

    if num_consts > 0:
        config()

        # TODO: do we really need to redefine this function instead of using the one outside?
        def eval_MSE(consts, num_variables: int):
            y_pred = eval_model(individual, Xs, D, num_variables, consts)
            total_err = np.sum((D.y-y_pred)**2)

            total_err *= 1/D.X.shape[0]
            return total_err

        # objective = partial(eval_MSE, num_variables=num_variables)
        objective = partial(jit(eval_MSE, static_argnums=(1,)),
                            num_variables=num_variables)

        obj_grad = partial(jit(grad(eval_MSE), static_argnums=(1,)),
                           num_variables=num_variables)
        x0 = np.ones(num_consts)

        class fitting_problem:
            def fitness(self, x):
                total_err = objective(x)
                # return [total_err + 0.*(np.linalg.norm(x, 2))**2]
                return [total_err]

            def gradient(self, x):
                return obj_grad(x)

            def get_bounds(self):
                return (-5.*np.ones(num_consts), 5.*np.ones(num_consts))

        # DCTKIT OPTIMIZATION INTERFACE
        # def get_bounds():
        #     return (-5.*np.ones(n_constants), 5.*np.ones(n_constants))

        # prb = oc.OptimizationProblem(
        #     dim=n_constants, state_dim=n_constants, objfun=objective)
        # prb.get_bounds = get_bounds
        # prb.set_obj_args({})
        # best_consts = prb.solve(x0, algo="lbfgs")
        # best_fit = objective(best_consts)

        # PYGMO SOLVER
        prb = pg.problem(fitting_problem())
        # algo = pg.algorithm(pg.scipy_optimize(method="BFGS"))
        algo = pg.algorithm(pg.nlopt(solver="lbfgs"))
        algo.extract(pg.nlopt).maxeval = 10
        # algo = pg.algorithm(pg.cmaes(gen=70))
        # algo = pg.algorithm(pg.pso(gen=70))
        # algo = pg.algorithm(pg.sea(gen=70))
        # pop = pg.population(prb, size=50)
        pop = pg.population(prb, size=1)
        pop.push_back(x0)
        pop = algo.evolve(pop)
        MSE = pop.champion_f[0]
        consts = pop.champion_x
    else:
        MSE = compute_MSE(individual, Xs, D, num_variables)
        consts = []
    if np.isinf(MSE) or np.isnan(MSE):
        MSE = 1e8/D.X.shape[0]
    return MSE, consts


def get_features_batch(individuals_str_batch,
                       individ_feature_extractors=[
                           len, check_nested_trig_fn,
                           check_trig_fn]):
    features_batch = [[fe(i) for i in individuals_str_batch]
                      for fe in individ_feature_extractors]

    individ_length = features_batch[0]
    nested_trigs = features_batch[1]
    num_trigs = features_batch[2]
    return individ_length, nested_trigs, num_trigs


@ray.remote(num_cpus=num_cpus)
def predict(individuals_str_batch,  toolbox, dataset, num_variables, penalty):
    if num_variables > 1:
        Xs = [dataset.X[:, i] for i in range(num_variables)]
    else:
        Xs = [dataset.X]

    predictions = [None]*len(individuals_str_batch)

    for i, tree in enumerate(individuals_str_batch):
        callable, _ = compile_individual_with_consts(tree, toolbox)
        predictions[i] = eval_model(
            callable, Xs, dataset, num_variables, consts=tree.consts)

    return predictions


@ray.remote(num_cpus=num_cpus)
def compute_MSEs(individuals_str_batch, toolbox, dataset, num_variables,
                 penalty):
    if num_variables > 1:
        Xs = [dataset.X[:, i] for i in range(num_variables)]
    else:
        Xs = [dataset.X]

    total_errs = [None]*len(individuals_str_batch)

    for i, tree in enumerate(individuals_str_batch):
        callable, _ = compile_individual_with_consts(tree, toolbox)
        total_errs[i] = compute_MSE(
            callable, Xs, dataset, num_variables, consts=tree.consts)

    return total_errs


@ray.remote(num_cpus=num_cpus)
def compute_attributes(individuals_str_batch, toolbox, dataset, num_variables,
                       penalty):

    attributes = []*len(individuals_str_batch)

    individ_length, nested_trigs, num_trigs = get_features_batch(
        individuals_str_batch)

    for i, tree in enumerate(individuals_str_batch):

        MSE, consts = eval_MSE_and_tune_constants(
            tree, toolbox, dataset, num_variables)

        # Tarpeian selection
        if individ_length[i] >= 80:
            fitness = (1e8,)
        else:
            # add penalty on length of the tree to promote simpler solutions
            fitness = (MSE + 100000 * nested_trigs[i] + 0.*num_trigs[i]
                       + penalty["reg_param"]*individ_length[i],)
        attributes.append({'consts': consts, 'fitness': fitness})
    return attributes


def assign_attributes(individuals, attributes):
    for ind, attr in zip(individuals, attributes):
        ind.consts = attr["consts"]
        ind.fitness.values = attr["fitness"]


def generate_dataset(problem="Nguyen-8"):
    np.random.seed(42)
    range_train = None
    num_variables = 1
    if problem == "Nguyen-1":
        range_train = (-1., 1., 20)
        range_test = (1., 3., 20)
        def func(x): return np.power(x, 3)+np.power(x, 2)+x
    elif problem == "Nguyen-2":
        range_train = (-1., 1., 20)
        range_test = (1., 3., 20)
        def func(x): return np.power(x, 4)+np.power(x, 3)+np.power(x, 2)+x
    elif problem == "Nguyen-3":
        range_train = (-1., 1., 20)
        range_test = (1., 3., 20)

        def func(x): return np.power(x, 5)+np.power(x, 4) + \
            np.power(x, 3)+np.power(x, 2)+x
    elif problem == "Nguyen-4":
        range_train = (-1., 1., 20)
        range_test = (1., 3., 20)
        def func(x): return np.power(x, 6)+np.power(x, 5)+np.power(x, 4) + \
            np.power(x, 3)+np.power(x, 2)+x
    elif problem == "Nguyen-5":
        range_train = (-1., 1., 20)
        range_test = (1., 3., 20)
        def func(x): return np.sin(x*x)*np.cos(x)-1.
    elif problem == "Nguyen-6":
        range_train = (-1., 1., 20)
        range_test = (1., 3., 20)
        def func(x): return np.sin(x)+np.sin(x+x*x)
    elif problem == "Nguyen-7":
        range_train = (1.e-3, 2., 20)
        range_test = (3., 5., 20)
        def func(x): return np.log(1+x)+np.log(x*x+1)
    elif problem == "Nguyen-8":
        range_train = (1.e-3, 4., 20)
        range_test = (4., 8., 20)
        func = np.sqrt
    elif problem == "Nguyen-9":
        num_variables = 2
        range_train = (0., 1., 20)
        range_test = (1., 3., 20)
        def func(x): return np.sin(x[:, 0])+np.sin(x[:, 1]*x[:, 1])
    elif problem == "Nguyen-10":
        num_variables = 2
        range_train = (0., 1., 20)
        range_test = (1., 3., 20)
        def func(x): return 2.*np.sin(x[:, 0])*np.cos(x[:, 1])
    elif problem == "Nguyen-11":
        num_variables = 2
        range_train = (0., 1., 20)
        range_test = (1., 3., 20)
        def func(x): return np.power(x[:, 0], x[:, 1])
    elif problem == "Nguyen-12":
        num_variables = 2
        range_train = (-3., 3., 20)
        range_test = (0., 1., 20)
        def func(x): return np.power(
            x[:, 0], 4)-np.power(x[:, 0], 3)+0.5*np.power(x[:, 1], 2)-x[:, 1]
    elif problem == "Nguyen-13":
        range_train = (-1., 1., 20)
        range_test = (0., 1., 20)
        def func(x): return 3.39*np.power(x, 3)+2.12*np.power(x, 2)+1.78*x

    # Nguyen datasets
    if num_variables > 1 and range_train is not None:
        X_train = np.empty((range_train[-1], num_variables))
        X_test = np.empty((range_test[-1], num_variables))
        for i in range(num_variables):
            X_train[:, i] = np.random.uniform(*range_train)
            X_test[:, i] = np.random.uniform(*range_test)
    elif num_variables == 1 and range_train is not None:
        X_train = np.random.uniform(*range_train)
        X_test = np.random.uniform(*range_test)

    if range_train is not None:
        y_train = func(X_train)
        y_test = func(X_test)
    else:
        # PMLB datasets
        X, y = fetch_data(problem, return_X_y=True)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2)
        num_variables = X.shape[1]

    return X_train, y_train, X_test, y_test, num_variables


def alpine_bench(problem="Nguyen-8"):
    if problem == "Nguyen-13" or problem == "strogatz_glider1":
        with open("bench_alpine_Nguyen13.yaml") as config_file:
            config_file_data = yaml.safe_load(config_file)
    elif problem == "227_cpu_small":
        with open("bench_alpine_227_cpu_small.yaml") as config_file:
            config_file_data = yaml.safe_load(config_file)
    else:
        with open("bench_alpine.yaml") as config_file:
            config_file_data = yaml.safe_load(config_file)

    # generate training and test datasets
    X_train, y_train, X_test, y_test, num_variables = generate_dataset(problem)

    if num_variables == 1:
        pset = gp.PrimitiveSetTyped("Main", [float], float)
        pset.renameArguments(ARG0="x")
    elif num_variables == 2:
        pset = gp.PrimitiveSetTyped("Main", [float, float], float)
        pset.renameArguments(ARG0="x")
        pset.renameArguments(ARG1="y")
    else:
        pset = gp.PrimitiveSetTyped("Main", [float]*num_variables, float)

    penalty = config_file_data["gp"]["penalty"]
    common_params = {'penalty': penalty, 'num_variables': num_variables}

    batch_size = 1000
    callback_func = assign_attributes

    if problem == "Nguyen-13" or problem == "strogatz_glider1" or problem == "227_cpu_small":
        batch_size = 10
        config_file_data["gp"]["penalty"]["reg_param"] = 0.0001
        pset.addTerminal(object, float, "a")

    # seed_ind_str = [
    #     "sub(add(sub(mul(mul(x, x), mul(x, x)), mul(mul(x, x), x)), mul(div(x, add(x, x)), mul(y, y))), y)"]
    seed_ind_str = [
        "add(add(mul(a, mul(mul(x, x), x)), mul(a, mul(x, x))), mul(a, x))"]

    import re
    # lambda ind: len(re.findall("cos", str(ind)))

    gpsr = gps.GPSymbolicRegressor(pset=pset,
                                   fitness=compute_attributes.remote,
                                   predict_func=predict.remote,
                                   error_metric=compute_MSEs.remote,
                                   validate=False,
                                   common_data=common_params,
                                   callback_func=callback_func,
                                   print_log=True, num_best_inds_str=1,
                                   config_file_data=config_file_data,
                                   save_best_individual=True,
                                   output_path="./",
                                   seed=None, batch_size=batch_size)

    train_data = Dataset("dataset", X_train, y_train)
    test_data = Dataset("dataset", X_test, y_test)

    tic = time.time()
    gpsr.fit(train_data)
    toc = time.time()

    if hasattr(gpsr.best, "consts"):
        print("Best parameters = ", gpsr.best.consts)

    print("Elapsed time = ", toc-tic)
    time_per_individual = (toc-tic)/(gpsr.NGEN *
                                     gpsr.NINDIVIDUALS*gpsr.num_islands)
    individuals_per_sec = 1/time_per_individual
    print("Time per individual = ", time_per_individual)
    print("Individuals per sec = ", individuals_per_sec)

    u_best = gpsr.predict(test_data)
    print(u_best)
    print(y_test)

    plt.figure()
    plt.plot(u_best)
    plt.plot(y_test, '+')
    plt.show()

    MSE = np.sum((u_best-y_test)**2)/len(u_best)
    r2 = r2_score(y_test, u_best)
    print("MSE on the test set = ", MSE)
    print("R^2 on the test set = ", r2)
    if MSE <= 1e-10 or (problem == "Nguyen-13" and MSE <= 1e-5):
        return 1.
    else:
        return 0.


if __name__ == "__main__":
    # problems = ["Nguyen-1", "Nguyen-2", "Nguyen-3", "Nguyen-4", "Nguyen-5",
    #             "Nguyen-6", "Nguyen-7", "Nguyen-8", "Nguyen-9", "Nguyen-10",
    #             "Nguyen-11", "Nguyen-12", "Nguyen-13"]

    problems = ["227_cpu_small"]

    ave_success_rate = 0.

    with open('bench_stats.txt', 'w') as file:
        for problem in problems:
            success = 0.
            for i in range(num_runs):
                print("Problem {prb}, RUN #{num}".format(prb=problem, num=i))
                success += alpine_bench(problem=problem)
            success_rate = success/num_runs*100
            ave_success_rate += success_rate/len(problems)
            str_to_print = problem + " " + str(success_rate)
            print(str_to_print, file=file, flush=True)
        print("Average success rate = ", ave_success_rate)
