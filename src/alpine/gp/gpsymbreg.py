from deap import algorithms, tools, gp, base, creator
from operator import attrgetter
import matplotlib.pyplot as plt
import numpy as np
from mpire.utils import make_single_arguments
import operator
from typing import Tuple, List, Dict


def get_primitives_strings(pset: gp.PrimitiveSetTyped, types: list) -> List[str]:
    """Extract a list containing the names of all the primitives.

    Args:
        pset: a PrimitiveSetTyped object.
        types: list of all the types used in pset.

    Returns:
        a list containing the names (str) of the primitives.
    """
    primitives_strings = []
    for type in types:
        # NOTE: pset.primitives is a dictionary
        current_primitives = [str(pset.primitives[type][i].name)
                              for i in range(len(pset.primitives[type]))]
        primitives_strings.extend(current_primitives)
    return primitives_strings


def load_config_data(config_file_data: Dict, pset: gp.PrimitiveSetTyped) -> Tuple[Dict, Dict]:
    GPproblem_settings = dict()
    GPproblem_extra = dict()
    GPproblem_run = dict()
    GPproblem_settings['NINDIVIDUALS'] = config_file_data["gp"]["NINDIVIDUALS"]
    GPproblem_settings['NGEN'] = config_file_data["gp"]["NGEN"]
    GPproblem_settings['CXPB'] = config_file_data["gp"]["CXPB"]
    GPproblem_settings['MUTPB'] = config_file_data["gp"]["MUTPB"]
    GPproblem_settings['frac_elitist'] = int(
        config_file_data["gp"]["frac_elitist"]*GPproblem_settings['NINDIVIDUALS'])
    GPproblem_settings['min_'] = config_file_data["gp"]["min_"]
    GPproblem_settings['max_'] = config_file_data["gp"]["max_"]
    GPproblem_settings['overlapping_generation'] = config_file_data["gp"]["overlapping_generation"]
    GPproblem_settings['parsimony_pressure'] = config_file_data["gp"]["parsimony_pressure"]
    GPproblem_settings['tournsize'] = config_file_data["gp"]["select"]["tournsize"]
    GPproblem_settings['stochastic_tournament'] = config_file_data["gp"]["select"]["stochastic_tournament"]

    individualCreator, toolbox = creator_toolbox_config(
        config_file=config_file_data, pset=pset)
    GPproblem_settings['toolbox'] = toolbox
    GPproblem_settings['individualCreator'] = individualCreator

    GPproblem_extra['penalty'] = config_file_data["gp"]["penalty"]
    GPproblem_extra['n_jobs'] = config_file_data["mp"]["n_jobs"]

    GPproblem_run['early_stopping'] = config_file_data["gp"]["early_stopping"]
    GPproblem_run['plot_best'] = config_file_data["plot"]["plot_best"]
    GPproblem_run['plot_best_genealogy'] = config_file_data["plot"]["plot_best_genealogy"]

    return GPproblem_settings, GPproblem_run, GPproblem_extra


def creator_toolbox_config(config_file: dict, pset: gp.PrimitiveSetTyped) -> Tuple[gp.PrimitiveTree, base.Toolbox]:
    # initialize toolbox and creator
    toolbox = base.Toolbox()
    creator.create("FitnessMin", base.Fitness, weights=(-1.0, ))
    creator.create("Individual",
                   gp.PrimitiveTree,
                   fitness=creator.FitnessMin)
    createIndividual = creator.Individual

    min_ = config_file["gp"]["min_"]
    max_ = config_file["gp"]["max_"]

    expr_mut_fun = config_file["gp"]["mutate"]["expr_mut"]
    expr_mut_kargs = eval(config_file["gp"]["mutate"]["expr_mut_kargs"])

    toolbox.register("expr_mut", eval(expr_mut_fun), **expr_mut_kargs)

    crossover_fun = config_file["gp"]["crossover"]["fun"]
    crossover_kargs = eval(config_file["gp"]["crossover"]["kargs"])

    mutate_fun = config_file["gp"]["mutate"]["fun"]
    mutate_kargs = eval(config_file["gp"]["mutate"]["kargs"])
    toolbox.register("mate", eval(crossover_fun), **crossover_kargs)
    toolbox.register("mutate",
                     eval(mutate_fun), **mutate_kargs)
    toolbox.decorate(
        "mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
    toolbox.decorate(
        "mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))

    toolbox.register("expr", gp.genHalfAndHalf,
                     pset=pset, min_=min_, max_=max_)
    toolbox.register("expr_pop",
                     gp.genHalfAndHalf,
                     pset=pset,
                     min_=min_,
                     max_=max_,
                     is_pop=True)
    toolbox.register("individual", tools.initIterate,
                     createIndividual, toolbox.expr)
    toolbox.register("individual_pop", tools.initIterate,
                     createIndividual, toolbox.expr_pop)
    toolbox.register("population", tools.initRepeat,
                     list, toolbox.individual_pop)
    toolbox.register("compile", gp.compile, pset=pset)

    return createIndividual, toolbox


class GPSymbRegProblem():
    def __init__(self,
                 pset,
                 NINDIVIDUALS=10,
                 NGEN=1,
                 CXPB=0.5,
                 MUTPB=0.2,
                 frac_elitist=0.,
                 overlapping_generation=False,
                 parsimony_pressure={'enabled': False,
                                     'fitness_first': True,
                                     'parsimony_size': 1.5},
                 tournsize=3,
                 stochastic_tournament={'enabled': False, 'prob': [0.7, 0.3]},
                 min_=1,
                 max_=2,
                 individualCreator=None,
                 toolbox=None):
        """Symbolic regression problem via GP.

            Args:
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
        self.NINDIVIDUALS = NINDIVIDUALS
        self.NGEN = NGEN
        self.train_fit_history = []
        self.CXPB = CXPB
        self.MUTPB = MUTPB
        self.pop = None
        # best validation score among all the populations
        self.bvp = None
        # best training score among all the populations
        self.tbtp = None

        self.overlapping_generation = overlapping_generation
        self.parsimony_pressure = parsimony_pressure
        self.tournsize = tournsize
        self.stochastic_tournament = stochastic_tournament

        # Elitism settings
        self.n_elitist = int(frac_elitist*self.NINDIVIDUALS)

        if individualCreator is None:
            self.__default_creator()
        else:
            self.createIndividual = individualCreator

        # If toolbox is not provided, initialize it with default values
        if toolbox is None:
            self.__default_toolbox(pset, min_, max_)
        else:
            self.toolbox = toolbox
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

        # Create history object to build the genealogy tree
        self.history = tools.History()

        self.plot_initialized = False
        self.fig_id = 0

    def __init_logbook(self, overfit_measure=False):
        # Initialize logbook to collect statistics
        self.logbook = tools.Logbook()
        # Headers of fields to be printed during log
        if overfit_measure:
            self.logbook.header = "gen", "evals", "fitness", "size", "valid"
            self.logbook.chapters["valid"].header = "overfit", "valid_fit", "valid_err"
        else:
            self.logbook.header = "gen", "evals", "fitness", "size", "valerr"
        self.logbook.chapters["fitness"].header = "min", "avg", "max", "std"
        self.logbook.chapters["size"].header = "min", "avg", "max", "std"

    def __default_creator(self):
        creator.create("FitnessMin", base.Fitness, weights=(-1.0, ))
        creator.create("Individual",
                       gp.PrimitiveTree,
                       fitness=creator.FitnessMin)
        self.createIndividual = creator.Individual

    def __default_toolbox(self, pset, min_, max_):
        # Register functions to create individuals and initialize population
        self.toolbox = base.Toolbox()
        self.toolbox.register("expr",
                              gp.genHalfAndHalf,
                              pset=pset,
                              min_=min_,
                              max_=max_)
        self.toolbox.register("expr_pop",
                              gp.genHalfAndHalf,
                              pset=pset,
                              min_=min_,
                              max_=max_,
                              is_pop=True)
        self.toolbox.register("individual", tools.initIterate,
                              self.createIndividual, self.toolbox.expr)
        self.toolbox.register("individual_pop", tools.initIterate,
                              self.createIndividual, self.toolbox.expr_pop)
        self.toolbox.register("population", tools.initRepeat,
                              list, self.toolbox.individual_pop)
        self.toolbox.register("compile", gp.compile, pset=pset)

        # Register selection with elitism operator
        self.toolbox.register("select", self.select_with_elitism)

        # Register mate and mutate operators
        self.toolbox.register("mate", gp.cxOnePoint)
        self.toolbox.register("expr_mut", gp.genGrow, min_=1, max_=3)
        self.toolbox.register("mutate",
                              gp.mutUniform,
                              expr=self.toolbox.expr_mut,
                              pset=pset)

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
        best = tools.selBest(self.pop, k=1)[0]
        valid_fit = self.toolbox.evaluate_val_fit(best)[0]
        valid_err = self.toolbox.evaluate_val_MSE(best)
        overfit = 0
        if overfit_measure:
            training_fit = best.fitness.values[0]
            overfit = self.__overfit_measure(training_fit, valid_fit)
        return overfit, valid_fit, valid_err

    def compute_statistics(self, pop, gen, evals, overfit_measure=False,
                           print_log=False):
        """Computes and prints statistics of a population."""

        # Compile statistics for the current population
        record = self.mstats.compile(pop)

        # Compute satistics related to the validation set
        overfit, valid_fit, valid_err = self.__compute_valid_stats(
            overfit_measure)

        # Record the statistics in the logbook
        if overfit_measure:
            record["valid"] = {"overfit": overfit,
                               "valid_fit": valid_fit,
                               "valid_err": valid_err}
            self.logbook.record(gen=gen, evals=evals, **record)
        else:
            self.logbook.record(gen=gen, evals=evals, valerr=valid_err, **record)

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
            return bestind + self.selStochasticTournament(individuals, n_tournament,
                                                          tournsize=self.tournsize,
                                                          prob=self.
                                                          stochastic_tournament['prob'])
        else:
            return bestind + tools.selTournament(individuals, n_tournament,
                                                 tournsize=self.tournsize)

    def selStochasticTournament(self, individuals, k, tournsize, prob,
                                fit_attr="fitness"):
        """Select the best individual among *tournsize* randomly chosen individuals, *k*
        times. The list returned contains references to the input *individuals*.

        Args:
            individuals: A list of individuals to select from.
            k: The number of individuals to select.
            tournsize: The number of individuals participating in each tournament.
            fit_attr: The attribute of individuals to use as selection criterion
        Returns:
            A list of selected individuals.
        """
        chosen = []
        for _ in range(k):
            aspirants = tools.selection.selRandom(individuals, tournsize)
            aspirants.sort(key=attrgetter(fit_attr), reverse=True)
            chosen_index = int(np.random.choice(range(tournsize), 1, p=prob))
            chosen.append(aspirants[chosen_index])
        return chosen

    def __plot_history(self):
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
        plt.plot(x, self.val_fit_history, 'r', label="Validation Fitness")
        # fig.legend(loc='upper right')
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

    def run(self,
            plot_history=False,
            print_log=False,
            plot_best=False,
            plot_freq=5,
            plot_best_genealogy=False,
            seed=None,
            n_splits=10,
            early_stopping={'enabled': False, 'max_overfit': 0},
            preprocess_fun=None,
            callback_fun=None):
        """Runs symbolic regression.

            Args:
                plot_history: whether to plot fitness vs generation number.
                seed: list of individuals to seed in the initial population.
                preprocess_fun: function to call before evaluating the fitness of the
                    individuals of each generation.
                callback_fun: function to call after evaluating the fitness of the
                    individuals of each generation.
        """
        if plot_best_genealogy:
            # Decorators for history
            self.toolbox.decorate("mate", self.history.decorator)
            self.toolbox.decorate("mutate", self.history.decorator)

        print("> MODEL TRAINING/SELECTION STARTED", flush=True)

        # Generate initial population
        print("Generating initial population...", flush=True)
        self.pop = self.toolbox.population(n=self.NINDIVIDUALS)

        if plot_best_genealogy:
            # Populate the history and the Hall Of Fame
            self.history.update(self.pop)

        # Initialize logbook for statistics
        self.__init_logbook(overfit_measure=early_stopping['enabled'])

        if seed is not None:
            print("Seeding population with individuals...", flush=True)
            self.pop[:len(seed)] = seed

        print(" -= START OF EVOLUTION =- ", flush=True)

        # Evaluate the fitness of the entire population on the training set
        print("Evaluating initial population...", flush=True)

        if preprocess_fun is not None:
            preprocess_fun(self.pop)

        fitnesses = self.toolbox.map(self.toolbox.evaluate_train,
                                     make_single_arguments(self.pop),
                                     iterable_len=self.NINDIVIDUALS,
                                     n_splits=n_splits)
        for ind, fit in zip(self.pop, fitnesses):
            ind.fitness.values = fit

        print("DONE.", flush=True)

        if early_stopping['enabled']:
            print("Using early-stopping.")
            best = tools.selBest(self.pop, k=1)[0]
            self.tbtp = best.fitness.values[0]
            # Evaluate fitness on the validation set
            self.bvp = self.toolbox.evaluate_val_fit(best)[0]
            self.best = best
            self.last_improvement = self.tbtp
            # initialize overfit index m
            m = 0
            # initialize last generation without overfitting
            self.last_gen_no_overfit = 0

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

            fitnesses = self.toolbox.map(self.toolbox.evaluate_train,
                                         make_single_arguments(invalid_ind),
                                         iterable_len=len(invalid_ind),
                                         n_splits=n_splits)

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
                                    overfit_measure=early_stopping['enabled'],
                                    print_log=print_log)

            print(f"The best individual of this generation is: {best}")

            if callback_fun is not None:
                callback_fun(self.pop)

            # Update history of best fitness and best validation error
            self.train_fit_history = self.logbook.chapters["fitness"].select("min")
            self.val_fit_history = self.logbook.chapters["valid"].select("valid_fit")
            if early_stopping['enabled']:
                self.val_fit_history = self.logbook.chapters["valid"].select(
                    "valid_fit")
                self.min_valerr = min(self.val_fit_history)
            else:
                self.val_fit_history = self.logbook.select("valerr")
                self.min_valerr = min(self.logbook.select("valerr"))

            if plot_history and (cgen % plot_freq == 0 or cgen == 1):
                self.__plot_history()

            if plot_best and (self.toolbox.plot_best_func is not None) \
                    and (cgen % plot_freq == 0 or cgen == 1):
                self.toolbox.plot_best_func(best)
                if cgen != self.NGEN and m != early_stopping:
                    plt.clf()

            if early_stopping['enabled']:
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

                if m == early_stopping['max_overfit']:
                    # save number of generations when stopping for the last training run
                    self.NGEN = self.last_gen_no_overfit
                    print("-= EARLY STOPPED =-")
                    break

        self.plot_initialized = False
        print(" -= END OF EVOLUTION =- ", flush=True)

        print("> MODEL TRAINING/SELECTION COMPLETED", flush=True)

        if plot_best_genealogy:
            self.__plot_genealogy(best)
