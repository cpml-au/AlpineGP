from deap import algorithms, tools, gp, base, creator
from operator import attrgetter
import matplotlib.pyplot as plt
import numpy as np
from mpire.utils import make_single_arguments


class GPSymbRegProblem():
    def __init__(self,
                 pset,
                 NINDIVIDUALS,
                 NGEN,
                 CXPB,
                 MUTPB,
                 frac_elitist=0.,
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
                frac_elitist: best individuals to keep expressed as a percentage of the
                population (ex. 0.1 = keep top 10% individuals)
        """
        self.pset = pset
        self.NINDIVIDUALS = NINDIVIDUALS
        self.NGEN = NGEN
        self.min_fit_history = []
        self.CXPB = CXPB
        self.MUTPB = MUTPB
        self.pop = None
        # best validation score among all the populations
        self.bvp = None
        # best training score among all the populations
        self.tbtp = None

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

        # Create history object to build the genealogy tree (NOT USED for now)
        self.history = tools.History()

        # Create Hall Of Fame object
        # self.halloffame = tools.HallOfFame(maxsize=5)

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
        self.toolbox.register("individual", tools.initIterate,
                              self.createIndividual, self.toolbox.expr)
        # creator_package = (self.createIndividual, self.toolbox.expr)
        # self.toolbox.register("population", generate_population, creator_package)
        self.toolbox.register("population", tools.initRepeat,
                              list, self.toolbox.individual)
        self.toolbox.register("compile", gp.compile, pset=pset)

        # Register selection with elitism operator
        self.toolbox.register("select", self.select_with_elitism)

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
        valid_fit = self.toolbox.evaluate_val_fit(best[0])[0]
        valid_err = self.toolbox.evaluate_val_MSE(best[0])
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
        """Select the best individual among *tournsize* randomly chosen
        individuals, *k* times. The list returned contains
        references to the input *individuals*.
        :param individuals: A list of individuals to select from.
        :param k: The number of individuals to select.
        :param tournsize: The number of individuals participating in each tournament.
        :param fit_attr: The attribute of individuals to use as selection criterion
        :returns: A list of selected individuals.
        This function uses the :func:`~random.choice` function from the python base
        :mod:`random` module.
        """
        chosen = []
        for _ in range(k):
            aspirants = tools.selection.selRandom(individuals, tournsize)
            aspirants.sort(key=attrgetter(fit_attr), reverse=True)
            chosen_index = int(np.random.choice(range(tournsize), 1, p=prob))
            chosen.append(aspirants[chosen_index])
        return chosen

    def run(self,
            plot_history=False,
            print_log=False,
            plot_best=False,
            plot_best_func=None,
            plot_freq=5,
            plot_best_genealogy=False,
            seed=None,
            n_splits=10,
            early_stopping={'enabled': False, 'max_overfit': 0}):
        """Runs symbolic regression.

            Args:
                seed: list of individuals to seed in the initial population.
        """
        if plot_best_genealogy:
            # Decorators for history
            self.toolbox.decorate("mate", self.history.decorator)
            self.toolbox.decorate("mutate", self.history.decorator)

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
        fitnesses = list(self.toolbox.map(self.toolbox.evaluate_train, make_single_arguments(
            self.pop), iterable_len=self.NINDIVIDUALS, n_splits=n_splits))
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
            offspring = list(
                map(self.toolbox.clone,
                    self.toolbox.select(self.pop)))

            # Apply crossover and mutation to the offspring, except elite individuals
            elite_ind = tools.selBest(offspring, self.n_elitist)
            offspring = elite_ind + \
                algorithms.varOr(offspring, self.toolbox, self.NINDIVIDUALS -
                                 self.n_elitist, self.CXPB, self.MUTPB)

            # Evaluate the individuals with an invalid fitness (subject to crossover or
            # mutation)
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = self.toolbox.map(self.toolbox.evaluate_train, make_single_arguments(
                invalid_ind), iterable_len=len(invalid_ind), n_splits=n_splits)

            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            # The population is entirely replaced by the offspring
            self.pop[:] = offspring

            # Select the best individual in the current population
            best = tools.selBest(self.pop, k=1)[0]

            # Compute population statistics
            self.compute_statistics(self.pop,
                                    cgen,
                                    len(invalid_ind),
                                    overfit_measure=early_stopping['enabled'],
                                    print_log=print_log)

            # Update history of best fitness and best validation error
            self.min_fit_history = self.logbook.chapters["fitness"].select("min")
            self.val_fit_history = self.logbook.chapters["valid"].select("valid_fit")
            if early_stopping['enabled']:
                self.val_fit_history = self.logbook.chapters["valid"].select(
                    "valid_fit")
                self.min_valerr = min(self.val_fit_history)
            else:
                self.val_fit_history = self.logbook.select("valerr")
                self.min_valerr = min(self.logbook.select("valerr"))

            if plot_history and (cgen % plot_freq == 0 or cgen == 1):
                if not self.plot_initialized:
                    self.plot_initialized = True
                    # new figure number when starting with new evolution
                    self.fig_id = self.fig_id + 1
                    plt.figure(self.fig_id).show()
                    plt.pause(0.01)

                plt.figure(self.fig_id)
                fig = plt.gcf()

                # Array of generations starts from 1
                x = range(1, len(self.min_fit_history) + 1)
                plt.plot(x, self.min_fit_history, 'b', label="Training Fitness")
                plt.plot(x, self.val_fit_history, 'r', label="Validation Fitness")
                # Plotting mean results often in very large numbers
                # plt.plot(self.mean_history, 'r')
                # plt.yscale('log')
                plt.xlabel("Generation #")
                plt.ylabel("Best Fitness")

                fig.canvas.draw()

                plt.pause(0.01)

            if plot_best and (plot_best_func
                              is not None) and (cgen % plot_freq == 0 or cgen == 1):
                best = tools.selBest(self.pop, k=1)
                plot_best_func(best[0])

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

                self.last_improvement = training_fit
                print(f"The best of this generation is: {best}")
                print(f"The best until now is: {self.best}")

                if m == early_stopping['max_overfit']:
                    # save number of generations when stopping for the last training run
                    self.NGEN = self.last_gen_no_overfit
                    print("-= EARLY STOPPED =-")
                    break

        self.plot_initialized = False
        print(" -= END OF EVOLUTION =- ", flush=True)

        if plot_best_genealogy:
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
