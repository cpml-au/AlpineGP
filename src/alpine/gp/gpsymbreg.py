from deap import algorithms, tools, gp, base, creator
import matplotlib.pyplot as plt
import numpy as np
import operator
from typing import List, Dict, Callable
from os.path import join
import networkx as nx
from .primitives import addPrimitivesToPset
import os
import ray

# reducing the number of threads launched by fitness evaluations
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'

os.environ["NUM_INTER_THREADS"] = "1"
os.environ["NUM_INTRA_THREADS"] = "1"

os.environ["XLA_FLAGS"] = ("--xla_cpu_multi_thread_eigen=false "
                           "intra_op_parallelism_threads=1")


class GPSymbRegProblem():
    def __init__(self,
                 pset: gp.PrimitiveSet | gp.PrimitiveSetTyped,
                 toolbox: base.Toolbox = None,
                 individualCreator: gp.PrimitiveTree = None,
                 NINDIVIDUALS: int = 10,
                 NGEN: int = 1,
                 CXPB: float = 0.5,
                 MUTPB: float = 0.2,
                 frac_elitist: float = 0.,
                 overlapping_generation: bool = False,
                 parsimony_pressure={'enabled': False,
                                     'fitness_first': True,
                                     'parsimony_size': 1.5},
                 tournsize: int = 3,
                 stochastic_tournament={'enabled': False, 'prob': [0.7, 0.3]},
                 config_file_data: Dict | None = None,
                 use_ray=True):
        """Symbolic regression problem via Genetic Programming.

            Args:
                pset: set of primitives and terminals.
                toolbox: Set to None if `config_file_data` is provided.
                individualCreator: Set to None if `config_file_data` is provided.
                NINDIVIDUALS: number of individuals in the parent population.
                NGEN: number of generations.
                CXPB: cross-over probability.
                MUTPB: mutation probability.
                frac_elitist: best individuals to keep expressed as a percentage of the
                    population (ex. 0.1 = keep top 10% individuals)
                overlapping_generation: True if the offspring competes with the parents
                    for survival.
        """
        self.pset = pset
        if config_file_data is not None:
            self.load_config_data(config_file_data)
        else:
            self.NINDIVIDUALS = NINDIVIDUALS
            self.NGEN = NGEN
            self.CXPB = CXPB
            self.MUTPB = MUTPB

            self.overlapping_generation = overlapping_generation
            self.parsimony_pressure = parsimony_pressure
            self.tournsize = tournsize
            self.stochastic_tournament = stochastic_tournament
            self.early_stopping = {'enabled': False, 'max_overfit': 0}

            # Elitism settings
            self.n_elitist = int(frac_elitist*self.NINDIVIDUALS)

            self.createIndividual = individualCreator

            self.toolbox = toolbox

        # FIXME: move this instruction in the initialization of the toolbox
        self.toolbox.register("select", self.select_with_elitism)

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

        self.data_store = dict()

        # best validation score among all the populations
        self.bvp = None
        # best training score among all the populations
        self.tbtp = None

        # Create history object to build the genealogy tree
        self.history = tools.History()

        self.use_ray = use_ray
        if use_ray:
            ray.init()

        self.plot_initialized = False
        self.fig_id = 0

    def __creator_toolbox_config(self, config_file_data: Dict):
        """Initialize toolbox and individual creator based on config file."""
        toolbox = base.Toolbox()
        creator.create("FitnessMin", base.Fitness, weights=(-1.0, ))

        min_ = config_file_data["gp"]["min_"]
        max_ = config_file_data["gp"]["max_"]

        expr_mut_fun = config_file_data["gp"]["mutate"]["expr_mut"]
        expr_mut_kargs = eval(config_file_data["gp"]["mutate"]["expr_mut_kargs"])

        toolbox.register("expr_mut", eval(expr_mut_fun), **expr_mut_kargs)

        crossover_fun = config_file_data["gp"]["crossover"]["fun"]
        crossover_kargs = eval(config_file_data["gp"]["crossover"]["kargs"])

        mutate_fun = config_file_data["gp"]["mutate"]["fun"]
        mutate_kargs = eval(config_file_data["gp"]["mutate"]["kargs"])
        toolbox.register("mate", eval(crossover_fun), **crossover_kargs)
        toolbox.register("mutate",
                         eval(mutate_fun), **mutate_kargs)
        toolbox.decorate(
            "mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
        toolbox.decorate(
            "mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))

        toolbox.register("expr", gp.genHalfAndHalf,
                         pset=self.pset, min_=min_, max_=max_)
        toolbox.register("expr_pop",
                         gp.genHalfAndHalf,
                         pset=self.pset,
                         min_=min_,
                         max_=max_,
                         is_pop=True)
        creator.create("Individual",
                       gp.PrimitiveTree,
                       fitness=creator.FitnessMin)
        createIndividual = creator.Individual
        toolbox.register("individual", tools.initIterate,
                         createIndividual, toolbox.expr)

        # toolbox.register("individual_pop", tools.initIterate,
        #                 createIndividual, toolbox.expr_pop)
        toolbox.register("population", tools.initRepeat,
                         list, toolbox.individual)
        toolbox.register("compile", gp.compile, pset=self.pset)

        self.toolbox = toolbox
        self.createIndividual = createIndividual

    def load_config_data(self, config_file_data: Dict):
        """Load problem settings from YAML file."""
        self.NINDIVIDUALS = config_file_data["gp"]["NINDIVIDUALS"]
        self.NGEN = config_file_data["gp"]["NGEN"]
        self.CXPB = config_file_data["gp"]["CXPB"]
        self.MUTPB = config_file_data["gp"]["MUTPB"]
        self.n_elitist = int(config_file_data["gp"]["frac_elitist"]*self.NINDIVIDUALS)
        self.overlapping_generation = config_file_data["gp"]["overlapping_generation"]
        self.parsimony_pressure = config_file_data["gp"]["parsimony_pressure"]
        self.tournsize = config_file_data["gp"]["select"]["tournsize"]
        self.stochastic_tournament = \
            config_file_data["gp"]["select"]["stochastic_tournament"]

        if len(config_file_data["gp"]['primitives']) == 0:
            addPrimitivesToPset(self.pset)
        else:
            addPrimitivesToPset(self.pset, config_file_data["gp"]['primitives'])

        self.__creator_toolbox_config(config_file_data=config_file_data)

        self.early_stopping = config_file_data["gp"]["early_stopping"]
        self.plot_best = config_file_data["plot"]["plot_best"]
        self.plot_best_genealogy = config_file_data["plot"]["plot_best_genealogy"]

    def store_eval_common_params(self, data: Dict):
        self.__store_eval_params('common', data)

    def store_eval_dataset_params(self, params_names: List[str],
                                  datasets: Dict):

        if not self.early_stopping['enabled'] and 'val' in datasets:
            for i, _ in enumerate(datasets['train']):
                try:
                    datasets['train'][i] = \
                        np.vstack((datasets['train'][i], datasets['val'][i]))
                # FIXME: this could be avoided if the dataset is appropriately
                # structured even in the case of a single sample
                except ValueError:
                    datasets['train'][i] = \
                        np.hstack((datasets['train'][i], datasets['val'][i]))

        for dataset_label in datasets.keys():
            keys_values = dict(zip(params_names, datasets[dataset_label]))
            self.__store_eval_params(dataset_label, keys_values)

    def __store_eval_params(self, label: str, data: Dict):
        for key, value in data.items():
            data[key] = ray.put(value)
        self.data_store[label] = data

    def set_eval_args(self):
        store = self.data_store
        if self.early_stopping['enabled']:
            self.args_val = store['common'] | store['val']
        self.args_train = store['common'] | store['train']
        self.args_test_MSE = store['common'] | store['test']
        self.args_test_sols = self.args_test_MSE
        self.args_set = True

    def register_eval_funcs(self, fitness: Callable,
                            error_metric: Callable | None = None,
                            test_sols: Callable | None = None):
        """Register functions for the evaluation of the fitness of the individuals over
        the datasets.
        """
        if not hasattr(self, "args_set") or not self.args_set:
            self.set_eval_args()

        if self.early_stopping['enabled']:
            self.toolbox.register(
                "evaluate_val_fit", fitness, **self.args_val)
            self.toolbox.register(
                "evaluate_val_MSE", error_metric, **self.args_val)

        self.toolbox.register("evaluate_train", fitness, **self.args_train)

        if error_metric is not None:
            self.toolbox.register("evaluate_test_MSE", error_metric,
                                  **self.args_test_MSE)
        if test_sols is not None:
            self.toolbox.register("evaluate_test_sols", test_sols,
                                  **self.args_test_sols)

    def __init_logbook(self, overfit_measure=False):
        # Initialize logbook to collect statistics
        self.logbook = tools.Logbook()
        # Headers of fields to be printed during log
        if overfit_measure:
            self.logbook.header = "gen", "evals", "fitness", "size", "valid"
            self.logbook.chapters["valid"].header = "overfit", "valid_fit", "valid_err"
        else:
            self.logbook.header = "gen", "evals", "fitness", "size"
        self.logbook.chapters["fitness"].header = "min", "avg", "max", "std"
        self.logbook.chapters["size"].header = "min", "avg", "max", "std"

    def __overfit_measure(self, training_fit, validation_fit):
        if (training_fit > validation_fit):
            overfit = 0
        elif (validation_fit < self.bvp):
            overfit = 0
            self.bvp = validation_fit
            self.tbtp = training_fit
        else:
            overfit = np.abs(training_fit - validation_fit) - \
                np.abs(self.tbtp - self.bvp)
        return overfit

    def __compute_valid_stats(self, overfit_measure=False):
        best = tools.selBest(self.pop, k=1)
        # FIXME: ugly way of handling lists/tuples; assume eval_val_MSE returns a
        # single-valued tuple as eval_val_fit
        valid_fit = self.toolbox.map(self.toolbox.evaluate_val_fit, best)[0][0]
        valid_err = self.toolbox.map(self.toolbox.evaluate_val_MSE, best)[0]
        overfit = 0
        if overfit_measure:
            training_fit = best[0].fitness.values[0]
            overfit = self.__overfit_measure(training_fit, valid_fit)
        return overfit, valid_fit, valid_err

    def compute_statistics(self, pop, gen, evals, overfit_measure=False,
                           print_log=False):
        """Computes and prints statistics of a population."""

        # Compile statistics for the current population
        record = self.mstats.compile(pop)

        # Record the statistics in the logbook
        if overfit_measure:
            # Compute satistics related to the validation set
            overfit, valid_fit, valid_err = self.__compute_valid_stats(overfit_measure)
            record["valid"] = {"overfit": overfit,
                               "valid_fit": valid_fit,
                               "valid_err": valid_err}

        self.logbook.record(gen=gen, evals=evals, **record)

        if print_log:
            # Print statistics for the current population
            print(self.logbook.stream, flush=True)

    def select_with_elitism(self, individuals):
        """Performs tournament selection with elitism.

            Args:
                individuals: a list of individuals to select from.

            Returns:
                population after selection/tournament.
        """
        n_tournament = self.NINDIVIDUALS - self.n_elitist

        bestind = tools.selBest(individuals, self.n_elitist)

        if self.parsimony_pressure['enabled']:
            return bestind + tools.selDoubleTournament(individuals, n_tournament,
                                                       fitness_size=n_tournament,
                                                       fitness_first=self.
                                                       parsimony_pressure
                                                       ['fitness_first'],
                                                       parsimony_size=self.
                                                       parsimony_pressure
                                                       ['parsimony_size'])

        if self.stochastic_tournament['enabled']:
            return bestind + tools.selStochasticTournament(individuals, n_tournament,
                                                           tournsize=self.tournsize,
                                                           prob=self.
                                                           stochastic_tournament['prob']
                                                           )
        else:
            return bestind + tools.selTournament(individuals, n_tournament,
                                                 tournsize=self.tournsize)

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
        if self.early_stopping['enabled']:
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

    def register_map(self, individ_feature_extractors: List[Callable] | None = None):
        def ray_mapper(f, individuals, toolbox):
            # Transform the tree expression in a callable function
            runnables = [toolbox.compile(expr=ind) for ind in individuals]
            feature_values = [[fe(i) for i in individuals]
                              for fe in individ_feature_extractors]
            fitnesses = ray.get([f(*args) for args in zip(runnables, *feature_values)])
            return fitnesses

        self.toolbox.register("map", ray_mapper, toolbox=self.toolbox)

    def run(self,
            plot_history: bool = False,
            print_log: bool = False,
            plot_best: bool | None = None,
            plot_freq: int = 5,
            plot_best_genealogy: bool | None = None,
            seed: List | None = None,
            preprocess_fun: Callable | None = None,
            callback_fun: Callable | None = None,
            plot_best_individual_tree: bool = True,
            print_best_test_MSE: bool = True,
            save_best_individual: bool = False,
            save_train_fit_history: bool = False,
            save_best_test_sols: bool = False,
            X_test_param_name: str | None = None,
            output_path: str | None = None):
        """Runs symbolic regression.

            Args:
                plot_history: whether to plot fitness vs generation number.
                print_log: whether to print the log containing the population statistics
                    during the run.
                plot_best: whether to show the plot of the solution corresponding to the
                    best individual every plot_freq generations.
                plot_freq: frequency (number of generations) of the plot of the best
                    individual.
                seed: list of individuals to seed in the initial population.
                preprocess_fun: function to call before evaluating the fitness of the
                    individuals of each generation.
                callback_fun: function to call after evaluating the fitness of the
                    individuals of each generation.
        """
        if plot_best is not None:
            self.plot_best = plot_best

        if plot_best_genealogy is not None:
            self.plot_best_genealogy = plot_best_genealogy

        if self.plot_best_genealogy:
            # Decorators for history
            self.toolbox.decorate("mate", self.history.decorator)
            self.toolbox.decorate("mutate", self.history.decorator)

        print("> MODEL TRAINING/SELECTION STARTED", flush=True)

        # Generate initial population
        print("Generating initial population...", flush=True)
        self.pop = self.toolbox.population(n=self.NINDIVIDUALS)

        print("DONE.", flush=True)

        if self.plot_best_genealogy:
            # Populate the history and the Hall Of Fame
            self.history.update(self.pop)

        # Initialize logbook for statistics
        self.__init_logbook(overfit_measure=self.early_stopping['enabled'])

        if seed is not None:
            print("Seeding population with individuals...", flush=True)
            self.pop[:len(seed)] = seed

        print(" -= START OF EVOLUTION =- ", flush=True)

        # Evaluate the fitness of the entire population on the training set
        print("Evaluating initial population...", flush=True)

        if preprocess_fun is not None:
            preprocess_fun(self.pop)

        fitnesses = self.toolbox.map(self.toolbox.evaluate_train, self.pop)

        for ind, fit in zip(self.pop, fitnesses):
            ind.fitness.values = fit

        if self.early_stopping['enabled']:
            print("Using early-stopping.")
            # TODO: these calculations seem to be repeating and could be grouped in a
            # function
            best = tools.selBest(self.pop, k=1)
            self.tbtp = best[0].fitness.values[0]
            # Evaluate fitness on the validation set
            self.bvp = self.toolbox.map(self.toolbox.evaluate_val_fit, best)[0][0]
            self.best = best[0]
            self.last_improvement = self.tbtp
            # initialize overfit index m
            m = 0
            # initialize last generation without overfitting
            self.last_gen_no_overfit = 0

        print("DONE.", flush=True)

        for gen in range(self.NGEN):
            cgen = gen + 1

            # Select and clone the next generation individuals
            offspring = list(map(self.toolbox.clone, self.toolbox.select(self.pop)))

            # Apply crossover and mutation to the offspring, except elite individuals
            elite_ind = tools.selBest(offspring, self.n_elitist)
            offspring = elite_ind + \
                algorithms.varOr(offspring, self.toolbox, self.NINDIVIDUALS -
                                 self.n_elitist, self.CXPB, self.MUTPB)

            # Evaluate the individuals with an invalid fitness (subject to crossover or
            # mutation)
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]

            if preprocess_fun is not None:
                preprocess_fun(invalid_ind)

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
            self.compute_statistics(self.pop,
                                    cgen,
                                    len(invalid_ind),
                                    overfit_measure=self.early_stopping['enabled'],
                                    print_log=print_log)

            print(f"The best individual of this generation is: {best}")

            if callback_fun is not None:
                callback_fun(self.pop)

            # Update history of best fitness and best validation error
            self.train_fit_history = self.logbook.chapters["fitness"].select("min")
            if self.early_stopping['enabled']:
                self.val_fit_history = self.logbook.chapters["valid"].select(
                    "valid_fit")
                self.val_fit_history = self.logbook.chapters["valid"].select(
                    "valid_fit")
                self.min_valerr = min(self.val_fit_history)

            if plot_history and (cgen % plot_freq == 0 or cgen == 1):
                self.__plot_history()

            if self.plot_best and (self.toolbox.plot_best_func is not None) \
                    and (cgen % plot_freq == 0 or cgen == 1 or cgen == self.NGEN):
                self.toolbox.plot_best_func(best)

            if self.early_stopping['enabled']:
                training_fit = best.fitness.values[0]
                # Retrieve last overtfit value
                overfit = self.logbook.chapters["valid"].select("overfit")[-1]
                if overfit == 0:
                    m = 0
                    self.last_gen_no_overfit = cgen
                    self.best = best
                elif np.abs(overfit) > 1e-3 and np.abs(self.last_improvement -
                                                       training_fit) >= 1e-1:
                    m += 1

                print(f"The best until now is: {self.best}")

                self.last_improvement = training_fit

                if m == self.early_stopping['max_overfit']:
                    self.NGEN = self.last_gen_no_overfit
                    print("-= EARLY STOPPED =-")
                    break

            else:
                self.best = best

        self.plot_initialized = False
        print(" -= END OF EVOLUTION =- ", flush=True)

        print("> MODEL TRAINING/SELECTION COMPLETED", flush=True)

        print(f"The best individual is {self.best}", flush=True)
        print(f"The best fitness on the training set is {self.train_fit_history[-1]}")

        if self.early_stopping['enabled']:
            print(f"The best fitness on the validation set is {self.min_valerr}")

        if self.plot_best_genealogy:
            self.__plot_genealogy(best)

        if plot_best_individual_tree:
            self.plot_best_individual_tree()

        if print_best_test_MSE and hasattr(self.toolbox, "evaluate_test_MSE"):
            self.print_best_test_MSE()

        if save_best_individual and output_path is not None:
            self.save_best_individual(output_path)
            print("String of the best individual saved to disk.")

        if save_train_fit_history and output_path is not None:
            self.save_train_fit_history(output_path)
            print("Training fitness history saved to disk.")

        if save_best_test_sols and output_path is not None and \
            hasattr(self.toolbox, "evaluate_test_sols")\
                and X_test_param_name is not None:
            self.save_best_test_sols(output_path, X_test_param_name)
            print("Best individual solution evaluated over the test set saved to disk.")

        if self.use_ray:
            # ray.shutdown()
            pass

    def plot_best_individual_tree(self):
        """Plots the tree of the best individual."""
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
        if self.early_stopping['enabled']:
            np.save(join(output_path, "val_fit_history.npy"), self.val_fit_history)

    def save_best_test_sols(self, output_path: str, X_test_param_name: str):
        """Saves solutions (.npy) corresponding to the best individual evaluated
        over the test dataset.
        """
        best_test_sols = self.toolbox.map(self.toolbox.evaluate_test_sols,
                                          (self.best,))[0]

        X_test = self.data_store['test'][X_test_param_name]
        if self.use_ray:
            X_test = ray.get(X_test)

        for i, sol in enumerate(best_test_sols):
            np.save(join(output_path, "best_sol_test_" + str(i) + ".npy"), sol)
            np.save(join(output_path, "true_sol_test_" + str(i) + ".npy"), X_test[i])

    def print_best_test_MSE(self):
        # map function always returns a list and takes a list as an input
        score_test = self.toolbox.map(self.toolbox.evaluate_test_MSE,
                                      (self.best,))[0]

        print(f"The best MSE on the test set is {score_test}")
