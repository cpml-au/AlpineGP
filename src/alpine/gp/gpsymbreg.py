from deap import algorithms, tools, gp, base, creator
import matplotlib.pyplot as plt
import numpy as np
import operator
from typing import List, Dict, Callable
from os.path import join
import networkx as nx
from .primitives import addPrimitivesToPset
from alpine.data import Dataset
import os
import ray
import random
from joblib import Parallel, delayed

# reducing the number of threads launched by fitness evaluations
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'

os.environ["NUM_INTER_THREADS"] = "1"
os.environ["NUM_INTRA_THREADS"] = "1"

os.environ["XLA_FLAGS"] = ("--xla_cpu_multi_thread_eigen=false "
                           "intra_op_parallelism_threads=1")


class GPSymbolicRegressor():
    """Symbolic regression problem via Genetic Programming.

        Args:
            pset: set of primitives and terminals (loosely or strongly typed).
            predict_func: function that returns a prediction given an individual and
                a test `Dataset` as inputs.
            toolbox: set to None if `config_file_data` is provided.
            individualCreator: set to None if `config_file_data` is provided.
            NINDIVIDUALS: number of individuals in the parent population.
            NGEN: number of generations.
            crossover_prob: cross-over probability.
            MUTPB: mutation probability.
            frac_elitist: best individuals to keep expressed as a percentage of the
                population (ex. 0.1 = keep top 10% individuals)
            overlapping_generation: True if the offspring competes with the parents
                for survival.
            plot_history: whether to plot fitness vs generation number.
            print_log: whether to print the log containing the population statistics
                during the run.
            plot_best: whether to show the plot of the solution corresponding to the
                best individual every plot_freq generations.
            plot_freq: frequency (number of generations) of the plot of the best
                individual.
            seed: list of individual strings to seed in the initial population.
            preprocess_func: function to call before evaluating the fitness of the
                individuals of each generation.
            callback_func: function to call after evaluating the fitness of the
                individuals of each generation.
    """

    def __init__(self,
                 pset: gp.PrimitiveSet | gp.PrimitiveSetTyped,
                 fitness: Callable,
                 error_metric: Callable | None = None,
                 predict_func: Callable | None = None,
                 common_data: Dict | None = None,
                 feature_extractors: List = [],
                 toolbox: base.Toolbox = None,
                 individualCreator: gp.PrimitiveTree = None,
                 NINDIVIDUALS: int = 10,
                 NGEN: int = 1,
                 crossover_prob: float = 0.5,
                 MUTPB: float = 0.2,
                 frac_elitist: float = 0.,
                 overlapping_generation: bool = False,
                 validate: bool = False,
                 preprocess_func: Callable | None = None,
                 callback_func: Callable | None = None,
                 seed: List[str] | None = None,
                 config_file_data: Dict | None = None,
                 plot_history: bool = False,
                 print_log: bool = False,
                 plot_best: bool = False,
                 plot_freq: int = 5,
                 plot_best_genealogy: bool = False,
                 plot_best_individual_tree: bool = False,
                 save_best_individual: bool = False,
                 save_train_fit_history: bool = False,
                 output_path: str | None = None,
                 parallel_lib: str = "ray",
                 parallel_backend: str = "threads",
                 num_jobs=-1):

        self.pset = pset

        self.fitness = fitness
        self.error_metric = error_metric
        self.predict_func = predict_func

        self.data_store = dict()

        self.plot_best = plot_best

        self.plot_best_genealogy = plot_best_genealogy

        self.plot_history = plot_history
        self.print_log = print_log
        self.plot_freq = plot_freq
        self.preprocess_func = preprocess_func
        self.callback_fun = callback_func
        self.is_plot_best_individual_tree = plot_best_individual_tree
        self.is_save_best_individual = save_best_individual
        self.is_save_train_fit_history = save_train_fit_history
        self.output_path = output_path
        self.parallel_lib = parallel_lib

        if common_data is not None:
            # FIXME: does everything work when the functions do not have common args?
            self.store_fit_error_common_args(common_data)

        if config_file_data is not None:
            self.__load_config_data(config_file_data)
        else:
            self.NINDIVIDUALS = NINDIVIDUALS
            self.NGEN = NGEN
            self.crossover_prob = crossover_prob
            self.MUTPB = MUTPB

            self.overlapping_generation = overlapping_generation
            self.validate = validate

            # Elitism settings
            self.n_elitist = int(frac_elitist*self.NINDIVIDUALS)

            self.createIndividual = individualCreator

            self.toolbox = toolbox

        self.seed = seed

        if self.seed is not None:
            self.seed = [self.createIndividual.from_string(i, pset) for i in seed]

        # Initialize variables for statistics
        self.stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
        self.stats_size = tools.Statistics(len)
        self.mstats = tools.MultiStatistics(fitness=self.stats_fit,
                                            size=self.stats_size)
        self.mstats.register("avg", lambda ind: np.around(np.mean(ind), 4))
        self.mstats.register("std", lambda ind: np.around(np.std(ind), 4))
        self.mstats.register("min", lambda ind: np.around(np.min(ind), 4))
        self.mstats.register("max", lambda ind: np.around(np.max(ind), 4))

        self.__init_logbook()

        self.train_fit_history = []

        # Create history object to build the genealogy tree
        self.history = tools.History()

        if self.plot_best_genealogy:
            # Decorators for history
            self.toolbox.decorate("mate", self.history.decorator)
            self.toolbox.decorate("mutate", self.history.decorator)

        if self.parallel_lib == "joblib":
            parallel = Parallel(n_jobs=num_jobs, prefer=parallel_backend)
        else:
            parallel = None
        self.__register_map(feature_extractors, parallel)

        self.plot_initialized = False
        self.fig_id = 0

    def __creator_toolbox_config(self, config_file_data: Dict):
        """Initialize toolbox and individual creator based on config file."""
        self.toolbox = base.Toolbox()

        # SELECTION
        select_fun = eval(config_file_data["gp"]["select"]["fun"])
        select_args = eval(config_file_data["gp"]["select"]["kargs"])
        self.toolbox.register("select", select_fun, **select_args)

        # MUTATION
        expr_mut_fun = config_file_data["gp"]["mutate"]["expr_mut"]
        expr_mut_kargs = eval(config_file_data["gp"]["mutate"]["expr_mut_kargs"])

        self.toolbox.register("expr_mut", eval(expr_mut_fun), **expr_mut_kargs)

        mutate_fun = config_file_data["gp"]["mutate"]["fun"]
        mutate_kargs = eval(config_file_data["gp"]["mutate"]["kargs"])

        self.toolbox.register("mutate",
                              eval(mutate_fun), **mutate_kargs)

        # CROSSOVER
        crossover_fun = config_file_data["gp"]["crossover"]["fun"]
        crossover_kargs = eval(config_file_data["gp"]["crossover"]["kargs"])

        self.toolbox.register("mate", eval(crossover_fun), **crossover_kargs)
        self.toolbox.decorate(
            "mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
        self.toolbox.decorate(
            "mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))

        # INDIVIDUAL GENERATOR/CREATOR
        min_ = config_file_data["gp"]["min_"]
        max_ = config_file_data["gp"]["max_"]
        self.toolbox.register("expr", gp.genHalfAndHalf,
                              pset=self.pset, min_=min_, max_=max_)
        self.toolbox.register("expr_pop",
                              gp.genHalfAndHalf,
                              pset=self.pset,
                              min_=min_,
                              max_=max_,
                              is_pop=True)
        creator.create("FitnessMin", base.Fitness, weights=(-1.0, ))
        creator.create("Individual",
                       gp.PrimitiveTree,
                       fitness=creator.FitnessMin)
        createIndividual = creator.Individual
        self.toolbox.register("individual", tools.initIterate,
                              createIndividual, self.toolbox.expr)

        self.toolbox.register("population", tools.initRepeat,
                              list, self.toolbox.individual)
        self.toolbox.register("compile", gp.compile, pset=self.pset)

        self.createIndividual = createIndividual

    def __load_config_data(self, config_file_data: Dict):
        """Load problem settings from YAML file."""
        self.NINDIVIDUALS = config_file_data["gp"]["NINDIVIDUALS"]
        self.NGEN = config_file_data["gp"]["NGEN"]
        self.crossover_prob = config_file_data["gp"]["crossover_prob"]
        self.MUTPB = config_file_data["gp"]["MUTPB"]
        self.n_elitist = int(config_file_data["gp"]["frac_elitist"]*self.NINDIVIDUALS)
        self.overlapping_generation = config_file_data["gp"]["overlapping_generation"]

        if len(config_file_data["gp"]['primitives']) == 0:
            addPrimitivesToPset(self.pset)
        else:
            addPrimitivesToPset(self.pset, config_file_data["gp"]['primitives'])

        self.__creator_toolbox_config(config_file_data=config_file_data)

        self.validate = config_file_data["gp"]["validate"]

    def store_fit_error_common_args(self, data: Dict):
        """Store names and values of the arguments that are in common between
        the fitness and the error metric functions in the common object space.

        Args:
            data: dictionary containing arguments names and values.
        """
        self.__store_shared_objects('common', data)

    def store_datasets(self, datasets: Dict[str, Dataset]):
        """Store datasets with the corresponding label ("train", "val" or "test")
        in the common object space. The datasets are passed as parameters to
        the fitness, and possibly to the error metric and the prediction functions.

        Args:
            datasets: the keys are 'train', 'val' and 'test' denoting the training,
                the validation and the test datasets, respectively. The associated
                values are `Dataset` objects.
        """
        for dataset_label in datasets.keys():
            dataset_name_data = {datasets[dataset_label].name: datasets[dataset_label]}
            self.__store_shared_objects(dataset_label, dataset_name_data)

    def __store_shared_objects(self, label: str, data: Dict):
        if self.parallel_lib == "ray":
            for key, value in data.items():
                data[key] = ray.put(value)
        self.data_store[label] = data

    def __init_logbook(self):
        # Initialize logbook to collect statistics
        self.logbook = tools.Logbook()
        # Headers of fields to be printed during log
        if self.validate:
            self.logbook.header = "gen", "evals", "fitness", "size", "valid"
            self.logbook.chapters["valid"].header = "valid_fit", "valid_err"
        else:
            self.logbook.header = "gen", "evals", "fitness", "size"
        self.logbook.chapters["fitness"].header = "min", "avg", "max", "std"
        self.logbook.chapters["size"].header = "min", "avg", "max", "std"

    def __compute_valid_stats(self):
        best = tools.selBest(self.pop, k=1)
        # FIXME: ugly way of handling lists/tuples; assume eval_val_MSE returns a
        # single-valued tuple as eval_val_fit
        valid_fit = self.toolbox.map(self.toolbox.evaluate_val_fit, best)[0][0]
        valid_err = self.toolbox.map(self.toolbox.evaluate_val_MSE, best)[0]

        return valid_fit, valid_err

    def __stats(self, pop, gen, evals):
        """Compute and print statistics of a population."""

        # Compile statistics for the current population
        record = self.mstats.compile(pop)

        # record the statistics in the logbook
        if self.validate:
            # compute satistics related to the validation set
            valid_fit, valid_err = self.__compute_valid_stats()
            record["valid"] = {"valid_fit": valid_fit,
                               "valid_err": valid_err}

        self.logbook.record(gen=gen, evals=evals, **record)

        if self.print_log:
            # Print statistics for the current population
            print(self.logbook.stream, flush=True)

    def tournament_with_elitism(self, individuals, tournsize=2,
                                stochastic_tournament={'enabled': False,
                                                       'prob': [1., 0.]}):
        """Perform tournament selection with elitism.

            Args:
                individuals: a list of individuals to select from.

            Returns:
                population after selection/tournament.
        """
        n_tournament = self.NINDIVIDUALS - self.n_elitist

        bestind = tools.selBest(individuals, self.n_elitist)

        if stochastic_tournament['enabled']:
            return bestind + tools.selStochasticTournament(individuals, n_tournament,
                                                           tournsize=tournsize,
                                                           prob=stochastic_tournament['prob']
                                                           )
        else:
            return bestind + tools.selTournament(individuals, n_tournament,
                                                 tournsize=tournsize)

    def __plot_history(self):
        """Plots the fitness of the best individual vs generation number."""
        if not self.plot_initialized:
            self.plot_initialized = True
            # new figure number when starting with new evolution
            self.fig_id = self.fig_id + 1
            plt.figure(self.fig_id).show()
            plt.pause(0.01)

        plt.figure(self.fig_id)
        fig = plt.gcf()

        # Array of generations starts from 1
        x = range(1, len(self.train_fit_history) + 1)
        plt.plot(x, self.train_fit_history, 'b', label="Training Fitness")
        if self.validate:
            plt.plot(x, self.val_fit_history, 'r', label="Validation Fitness")
            fig.legend(loc='upper right')

        plt.xlabel("Generation #")
        plt.ylabel("Best Fitness")

        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.pause(0.1)

    def __plot_genealogy(self, best):
        # Get genealogy of best individual
        import networkx
        gen_best = self.history.getGenealogy(best)
        graph = networkx.DiGraph(gen_best)
        graph = graph.reverse()
        pos = networkx.nx_agraph.graphviz_layout(
            graph, prog="dot", args="-Gsplines=True")
        # Retrieve individual strings for graph node labels
        labels = gen_best.copy()
        for key in labels.keys():
            labels[key] = str(self.history.genealogy_history[key])
        plt.figure()
        networkx.draw_networkx(graph, pos=pos)
        label_options = {"ec": "k", "fc": "lightblue", "alpha": 1.0}
        networkx.draw_networkx_labels(
            graph, pos=pos, labels=labels, font_size=10, bbox=label_options)

        # Save genealogy to file
        # networkx.nx_agraph.write_dot(graph, "genealogy.dot")

    def __register_fitness_func(self):
        store = self.data_store
        args_train = store['common'] | store['train']
        self.toolbox.register("evaluate_train", self.fitness, **args_train)

    def __register_val_funcs(self):
        """Register the functions needed for validation, i.e. the error metric and the
        fitness function. Must be called after storing the datasets in the common
        obj space.
        """
        store = self.data_store
        args_val = store['common'] | store['val']
        self.toolbox.register("evaluate_val_fit", self.fitness, **args_val)
        self.toolbox.register("evaluate_val_MSE", self.error_metric, **args_val)

    def __register_score_func(self):
        store = self.data_store
        args_score_func = store['common'] | store['test']
        self.toolbox.register("evaluate_test_score",
                              self.error_metric, **args_score_func)

    def __register_predict_func(self):
        store = self.data_store
        args_predict_func = store['common'] | store['test']
        self.toolbox.register("evaluate_test_sols",
                              self.predict_func, **args_predict_func)

    def __register_map(self, individ_feature_extractors: List[Callable] | None = None, parallel=None):
        def ray_mapper(f, individuals, toolbox):
            # Transform the tree expression in a callable function
            runnables = [toolbox.compile(expr=ind) for ind in individuals]
            feature_values = [[fe(i) for i in individuals]
                              for fe in individ_feature_extractors]
            if self.parallel_lib == "ray":
                fitnesses = ray.get([f(*args)
                                    for args in zip(runnables, *feature_values)])
            elif self.parallel_lib == "joblib":
                fitnesses = parallel((delayed(f)(*args)
                                      for args in zip(runnables,
                                                      *feature_values)))
            return fitnesses

        self.toolbox.register("map", ray_mapper, toolbox=self.toolbox)

    # @profile
    def fit(self, train_data: Dataset, val_data: Dataset | None = None):
        """Fits the training data using GP-based symbolic regression."""
        if self.validate and val_data is not None:
            datasets = {'train': train_data, 'val': val_data}
        else:
            datasets = {'train': train_data}
        self.store_datasets(datasets)
        self.__register_fitness_func()
        if self.validate and self.error_metric is not None:
            self.__register_val_funcs()
        self.__run()

    def predict(self, test_data: Dataset):
        datasets = {'test': test_data}
        self.store_datasets(datasets)
        self.__register_predict_func()
        u_best = self.toolbox.map(self.toolbox.evaluate_test_sols, (self.best,))[0]
        return u_best

    def score(self, test_data: Dataset):
        """Computes the error metric (passed to the `GPSymbolicRegressor` constructor)
            on a given dataset.
        """
        datasets = {'test': test_data}
        self.store_datasets(datasets)
        self.__register_score_func()
        score = self.toolbox.map(self.toolbox.evaluate_test_score, (self.best,))[0]
        return score

    def immigration(self, n_immigrants: int):
        immigrants = self.toolbox.population(n=n_immigrants)
        for i in range(n_immigrants):
            idx_individual_to_replace = random.randint(0, self.NINDIVIDUALS - 1)
            self.pop[idx_individual_to_replace] = immigrants[i]

    # @profile
    def __run(self):
        """Runs symbolic regression."""

        print("> MODEL TRAINING/SELECTION STARTED", flush=True)

        # Generate initial population
        print("Generating initial population...", flush=True)
        self.pop = self.toolbox.population(n=self.NINDIVIDUALS)

        print("DONE.", flush=True)

        if self.plot_best_genealogy:
            # Populate the history and the Hall Of Fame
            self.history.update(self.pop)

        if self.seed is not None:
            print("Seeding population with individuals...", flush=True)
            self.pop[:len(self.seed)] = self.seed

        print(" -= START OF EVOLUTION =- ", flush=True)

        # Evaluate the fitness of the entire population on the training set
        print("Evaluating initial population...", flush=True)

        if self.preprocess_func is not None:
            self.preprocess_func(self.pop)

        fitnesses = self.toolbox.map(self.toolbox.evaluate_train, self.pop)

        for ind, fit in zip(self.pop, fitnesses):
            ind.fitness.values = fit

        if self.validate:
            print("Using validation dataset.")

        print("DONE.", flush=True)

        for gen in range(self.NGEN):
            cgen = gen + 1

            if cgen % 10 == 0:
                print("Immigration.")
                self.immigration(990)

            # Select and clone the next generation individuals
            offspring = list(map(self.toolbox.clone, self.toolbox.select(self.pop)))

            # Apply crossover and mutation to the offspring, except elite individuals
            elite_ind = tools.selBest(offspring, self.n_elitist)
            offspring = elite_ind + \
                algorithms.varOr(offspring, self.toolbox, self.NINDIVIDUALS -
                                 self.n_elitist, self.crossover_prob, self.MUTPB)

            # Evaluate the individuals with an invalid fitness (subject to crossover or
            # mutation)
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]

            if self.preprocess_func is not None:
                self.preprocess_func(invalid_ind)

            fitnesses = self.toolbox.map(self.toolbox.evaluate_train, invalid_ind)

            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            if not self.overlapping_generation:
                # The population is entirely replaced by the offspring
                self.pop[:] = offspring
            else:
                # parents and offspring compete for survival (truncation selection)
                self.pop = tools.selBest(self.pop + offspring, self.NINDIVIDUALS)

            # select the best individual in the current population
            best = tools.selBest(self.pop, k=1)[0]
            # compute and print population statistics
            self.__stats(self.pop, cgen, len(invalid_ind))

            print(f"The best individual of this generation is: {best}")

            if self.callback_fun is not None:
                self.callback_fun(self.pop)

            # Update history of best fitness and best validation error
            self.train_fit_history = self.logbook.chapters["fitness"].select("min")
            if self.validate:
                self.val_fit_history = self.logbook.chapters["valid"].select(
                    "valid_fit")
                self.val_fit_history = self.logbook.chapters["valid"].select(
                    "valid_fit")
                self.min_valerr = min(self.val_fit_history)

            if self.plot_history and (cgen % self.plot_freq == 0 or cgen == 1):
                self.__plot_history()

            if self.plot_best and (self.toolbox.plot_best_func is not None) \
                    and (cgen % self.plot_freq == 0 or cgen == 1 or cgen == self.NGEN):
                self.toolbox.plot_best_func(best)

            self.best = best

        self.plot_initialized = False
        print(" -= END OF EVOLUTION =- ", flush=True)

        print("> MODEL TRAINING/SELECTION COMPLETED", flush=True)

        print(f"The best individual is {self.best}", flush=True)
        print(f"The best fitness on the training set is {self.train_fit_history[-1]}")

        if self.validate:
            print(f"The best fitness on the validation set is {self.min_valerr}")

        if self.plot_best_genealogy:
            self.__plot_genealogy(best)

        if self.is_plot_best_individual_tree:
            self.plot_best_individual_tree()

        if self.is_save_best_individual and self.output_path is not None:
            self.save_best_individual(self.output_path)
            print("String of the best individual saved to disk.")

        if self.is_save_train_fit_history and self.output_path is not None:
            self.save_train_fit_history(self.output_path)
            print("Training fitness history saved to disk.")

        # NOTE: ray.shutdown should be manually called by the user

    def plot_best_individual_tree(self):
        """Plots the tree of the best individual at the end of the evolution."""
        nodes, edges, labels = gp.graph(self.best)
        graph = nx.Graph()
        graph.add_nodes_from(nodes)
        graph.add_edges_from(edges)
        pos = nx.nx_agraph.graphviz_layout(graph, prog="dot")
        plt.figure(figsize=(7, 7))
        nx.draw_networkx_nodes(graph, pos, node_size=900, node_color="w")
        nx.draw_networkx_edges(graph, pos)
        nx.draw_networkx_labels(graph, pos, labels)
        plt.axis("off")
        plt.show()

    def save_best_individual(self, output_path: str):
        """Saves the string of the best individual of the population in a .txt file.
        """
        file = open(join(output_path, "best_ind.txt"), "w")
        file.write(str(self.best))
        file.close()

    def save_train_fit_history(self, output_path: str):
        np.save(join(output_path, "train_fit_history.npy"),
                self.train_fit_history)
        if self.validate:
            np.save(join(output_path, "val_fit_history.npy"), self.val_fit_history)

    def save_best_test_sols(self, test_data: Dataset, output_path: str):
        """Compute and save the predictions corresponding to the best individual
        at the end of the evolution, evaluated over the test dataset.

        Args:
            test_data: test dataset.
            output_path: path where the predictions should be saved (one .npy file for
                each sample in the test dataset).
        """
        best_test_sols = self.predict(test_data)

        for i, sol in enumerate(best_test_sols):
            np.save(join(output_path, "best_sol_test_" + str(i) + ".npy"), sol)

        print("Best individual solution evaluated over the test set saved to disk.")
