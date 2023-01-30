from deap import algorithms, tools, gp, base, creator
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
                 tournsize=3,
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
        self.min_history = []
        self.CXPB = CXPB
        self.MUTPB = MUTPB
        self.pop = None
        self.overfit = None
        self.bvp = None
        self.tbtp = None

        self.tournsize = tournsize

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
        self.hof = tools.HallOfFame(1)

        self.stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
        self.stats_size = tools.Statistics(len)
        self.mstats = tools.MultiStatistics(fitness=self.stats_fit,
                                            size=self.stats_size)
        self.mstats.register("avg", np.mean)
        self.mstats.register("std", np.std)
        self.mstats.register("min", np.min)
        self.mstats.register("max", np.max)

        # Initialize logbook to collect statistics
        self.logbook = tools.Logbook()
        # Headers of fields to be printed during log
        self.logbook.header = "gen", "evals", "fitness", "size"
        self.logbook.chapters["fitness"].header = "min", "avg", "max", "std"
        self.logbook.chapters["size"].header = "min", "avg", "max", "std"

        # Create history object to build the genealogy tree
        self.history = tools.History()

        # Create Hall Of Fame object
        self.halloffame = tools.HallOfFame(maxsize=5)

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
        self.toolbox.register("population", tools.initRepeat, list,
                              self.toolbox.individual)
        self.toolbox.register("compile", gp.compile, pset=pset)

        # Register selection with elitism operator
        self.toolbox.register("select", self.selElitistAndTournament)

    def __overfitting_measure(self, training_fit, validation_fit):
        if (training_fit > validation_fit):
            overfit = 0
        elif (validation_fit < self.bvp):
            overfit = 0
            self.bvp = validation_fit
            self.tbtp = training_fit
        else:
            overfit = np.abs(training_fit - validation_fit) - \
                np.abs(self.tbtp - self.bvp)
        self.overfit = overfit

    def compute_statistics(self, pop, gen, evals, print_log=False):
        """Computes and prints statistics of a population."""

        # Compile statistics for the current population
        record = self.mstats.compile(pop)

        # Record the statistics in the logbook
        self.logbook.record(gen=gen, evals=evals, **record)

        if print_log:
            # Print statistics for the current population
            print(self.logbook.stream, flush=True)

    def selElitistAndTournament(self, individuals):
        """Performs tournament selection with elitism.

            Args:
                individuals: a list of individuals to select from.

            Returns:
                population after selection/tournament.
        """
        n_tournament = self.NINDIVIDUALS - self.n_elitist

        return tools.selBest(individuals, self.n_elitist) + tools.selTournament(individuals, n_tournament, tournsize=self.tournsize)

    def run(self,
            plot_history=False,
            print_log=False,
            plot_best=False,
            plot_best_func=None,
            plot_freq=5,
            seed=None,
            n_splits=10,
            early_stopping=(False, 0)):
        """Runs symbolic regression.

            Args:
                seed: list of individuals to seed in the initial population.
        """

        # Generate initial population
        print("Generating initial population...", flush=True)
        self.pop = self.toolbox.population(n=self.NINDIVIDUALS)

        # Populate the history and the Hall Of Fame
        self.history.update(self.pop)
        self.halloffame.update(self.pop)

        if seed is not None:
            print("Seeding population with individuals...", flush=True)
            for i in range(len(seed)):
                self.pop[i] = seed[i]

        print(" -= START OF EVOLUTION =- ", flush=True)

        # Evaluate the entire population
        print("Evaluating initial population...", flush=True)
        fitnesses = list(self.toolbox.map(self.toolbox.evaluate, make_single_arguments(
            self.pop), iterable_len=self.NINDIVIDUALS, n_splits=n_splits))
        for ind, fit in zip(self.pop, fitnesses):
            ind.fitness.values = fit

        print("DONE.", flush=True)

        if early_stopping[0]:
            print("Using early-stopping.")
            best = tools.selBest(self.pop, k=1)
            self.tbtp = best[0].fitness.values[0]
            self.bvp = self.toolbox.evaluate_val(best[0])[0]
            self.best = best[0]
            self.last_improvement = self.tbtp
            # initialize m
            m = 0

        for gen in range(self.NGEN):
            cgen = gen + 1

            if early_stopping[0]:
                if m == early_stopping[1]:
                    self.pop = self.best
                    break

            # Select and clone the next generation individuals
            offspring = list(
                map(self.toolbox.clone,
                    self.toolbox.select(self.pop)))

            # Apply crossover and mutation to the offspring, except elite individuals
            offspring = tools.selBest(offspring, self.n_elitist) + algorithms.varOr(
                offspring, self.toolbox, self.NINDIVIDUALS - self.n_elitist, self.CXPB, self.MUTPB)

            # Evaluate the individuals with an invalid fitness (subject to crossover or
            # mutation)
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = self.toolbox.map(
                self.toolbox.evaluate, make_single_arguments(invalid_ind), iterable_len=len(invalid_ind), n_splits=n_splits)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            # The population is entirely replaced by the offspring
            self.pop[:] = offspring

            if early_stopping[0]:
                best = tools.selBest(self.pop, k=1)
                training_fit = best[0].fitness.values[0]
                valid_fit = self.toolbox.evaluate_val(best[0])[0]
                self.__overfitting_measure(training_fit, valid_fit)
                print(f"The current overfit measure is {self.overfit}")
                if self.overfit == 0:
                    m = 0
                    self.best = best[0]
                elif np.abs(self.overfit) > 1e-3 and np.abs(self.last_improvement - training_fit) >= 1e-1:
                    m += 1

                self.last_improvement = training_fit

                print(f"The current validation error is {valid_fit}")

            # Update Hall Of Fame
            self.halloffame.update(self.pop)
            self.compute_statistics(self.pop,
                                    cgen,
                                    len(invalid_ind),
                                    print_log=print_log)

            self.min_history = self.logbook.chapters["fitness"].select("min")

            if plot_history and cgen % plot_freq == 0:
                plt.figure(1)

                # Array of generations starts from 1
                x = range(1, len(self.min_history) + 1)
                plt.plot(x, self.min_history, 'b')
                # Plotting mean results often in very large numbers
                # plt.plot(self.mean_history, 'r')
                # plt.yscale('log')
                plt.xlabel("Generation #")
                plt.ylabel("Best Fitness")
                plt.draw()
                plt.pause(0.02)

            if plot_best and (plot_best_func
                              is not None) and cgen % plot_freq == 0:
                best = tools.selBest(self.pop, k=1)
                plot_best_func(best[0])

        print(" -= END OF EVOLUTION =- ", flush=True)
