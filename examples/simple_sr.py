import yaml
import os
from deap import gp
from alpine.gp.gpsymbreg import GPSymbolicRegressor
from alpine.data import Dataset
import numpy as np
import ray
import warnings
import re
from alpine.gp import util


def compile_individuals(toolbox, individuals_str_batch):
    return [toolbox.compile(expr=ind) for ind in individuals_str_batch]


# Ground truth
x = np.array([x / 10.0 for x in range(-10, 10)])
y = x**4 + x**3 + x**2 + x


def check_trig_fn(ind):
    return len(re.findall("cos", str(ind))) + len(re.findall("sin", str(ind)))


def check_nested_trig_fn(ind):
    return util.detect_nested_trigonometric_functions(str(ind))


def get_features_batch(
    individuals_str_batch,
    individ_feature_extractors=[len, check_nested_trig_fn, check_trig_fn],
):
    features_batch = [
        [fe(i) for i in individuals_str_batch] for fe in individ_feature_extractors
    ]

    individ_length = features_batch[0]
    nested_trigs = features_batch[1]
    num_trigs = features_batch[2]
    return individ_length, nested_trigs, num_trigs


def eval_MSE_sol(individual, true_data):
    warnings.filterwarnings("ignore")

    y_pred = individual(true_data.X)
    MSE = np.mean(np.square(y_pred - true_data.y))
    if np.isnan(MSE):
        MSE = 1e5
    return MSE, y_pred


@ray.remote
def predict(individuals_str, toolbox, true_data, penalty):

    callables = compile_individuals(toolbox, individuals_str)

    u = [None] * len(individuals_str)

    for i, ind in enumerate(callables):
        _, u[i] = eval_MSE_sol(ind, true_data)

    return u


@ray.remote
def score(individuals_str, toolbox, true_data, penalty):

    callables = compile_individuals(toolbox, individuals_str)

    MSE = [None] * len(individuals_str)

    for i, ind in enumerate(callables):
        MSE[i], _ = eval_MSE_sol(ind, true_data)

    return MSE


@ray.remote
def fitness(individuals_str, toolbox, true_data, penalty):
    callables = compile_individuals(toolbox, individuals_str)

    individ_length, nested_trigs, num_trigs = get_features_batch(individuals_str)

    fitnesses = [None] * len(individuals_str)
    for i, ind in enumerate(callables):
        if individ_length[i] >= 50:
            fitnesses[i] = (1e8,)
        else:
            MSE, _ = eval_MSE_sol(ind, true_data)

            fitnesses[i] = (
                MSE
                + 100000 * nested_trigs[i]
                + penalty["reg_param"] * individ_length[i],
            )

    return fitnesses


def main():
    yamlfile = "simple_sr.yaml"
    filename = os.path.join(os.path.dirname(__file__), yamlfile)
    with open(filename) as config_file:
        config_file_data = yaml.safe_load(config_file)

    pset = gp.PrimitiveSetTyped(
        "MAIN",
        [
            float,
        ],
        float,
    )
    pset.renameArguments(ARG0="x")

    penalty = config_file_data["gp"]["penalty"]
    common_data = {"penalty": penalty}

    gpsr = GPSymbolicRegressor(
        pset=pset,
        fitness=fitness.remote,
        error_metric=score.remote,
        predict_func=predict.remote,
        common_data=common_data,
        config_file_data=config_file_data,
        print_log=True,
        batch_size=100,
    )

    train_data = Dataset("true_data", x, y)
    gpsr.fit(train_data)

    ray.shutdown()


if __name__ == "__main__":
    main()
