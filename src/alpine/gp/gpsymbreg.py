from deap import algorithms, tools, gp, base, creator
import matplotlib.pyplot as plt
import numpy as np
from memory_profiler import profile


class GPSymbRegProblem():
    def __init__(self,
                 pset,
                 NINDIVIDUALS,
                 NGEN,
                 CXPB,
                 MUTPB,
                 min_=1,
                 max_=2,
                 individualCreator=None,
                 toolbox=None,
                 best_individuals=None):

        self.pset = pset
        self.NINDIVIDUALS = NINDIVIDUALS
        self.NGEN = NGEN
        self.min_history = []
        self.CXPB = CXPB
        self.MUTPB = MUTPB
        self.pop = None
        self.best_individuals = best_individuals

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

    def compute_statistics(self, pop, gen, evals, print_log=False):
        """Computes and prints statistics (max, min, avg, std) of a population."""

        # Compile statistics for the current population
        record = self.mstats.compile(pop)

        # Record the statistics in the logbook
        self.logbook.record(gen=gen, evals=evals, **record)

        if print_log:
            # Print statistics for the current population
            print(self.logbook.stream)

    def selElitistAndTournament(self, individuals, frac_elitist, tournsize=3):
        """Performs tournament selection with elitism.

            Args:
                individuals: a list of individuals to select from.
                frac_elitist: best individuals to keep expressed as a percentage of the population (ex. 0.1 = keep top 10% individuals)
                tournsize: tournament size.

            Returns:
                population after selection/tournament.
        """
        n_elitist = int(frac_elitist*self.NINDIVIDUALS)
        n_tournament = self.NINDIVIDUALS - n_elitist

        return tools.selBest(individuals, n_elitist) + tools.selTournament(individuals, n_tournament, tournsize=tournsize)

    # @profile
    def run(self,
            plot_history=False,
            print_log=False,
            plot_best=False,
            plot_best_func=None,
            plot_freq=5,
            seed=None):
        """Runs symbolic regression."""

        # Generate initial population
        print("Generating initial population...")
        self.pop = self.toolbox.population(n=self.NINDIVIDUALS)
        if self.best_individuals is not None:
            for i in range(len(self.best_individuals)):
                # print(str(self.best_individuals[i]))
                # print(type(self.best_individuals[i]))
                # print(type(self.pop[i]))
                self.pop[i] = self.best_individuals[i]
                # init_individ = "Add(MulFloat(Inner1(CoboundaryP0(u0), CoboundaryP0(u0)), 1/2), Inner0(u0, u1))"
                # self.pop[0] = creator.Individual.from_string(init_individ, self.pset)
                # print(len(self.pop[0]))

                # for i in range(self.NINDIVIDUALS):
                #    print(self.pop[i])

                # Populate the history and the Hall Of Fame
        self.history.update(self.pop)
        self.halloffame.update(self.pop)

        if seed is not None:
            print("Seeding population with individuals...")
            seedTree = gp.PrimitiveTree.from_string(seed, self.pset)
            self.pop[-1] = self.createIndividual(seedTree)

        print(" -= START OF EVOLUTION =- ")

        # Evaluate the entire population
        print("Evaluating initial population...")
        fitnesses = list(self.toolbox.map(self.toolbox.evaluate, self.pop))
        for ind, fit in zip(self.pop, fitnesses):
            ind.fitness.values = fit
        print("DONE.")

        for gen in range(self.NGEN):
            cgen = gen + 1

            # Select and clone the next generation individuals
            offspring = list(
                map(self.toolbox.clone,
                    # self.toolbox.select(self.pop, len(self.pop))))
                    self.toolbox.select(self.pop)))

            # Apply crossover and mutation to the offspring (like eaSimple)
            offspring = algorithms.varOr(
                offspring, self.toolbox, self.NINDIVIDUALS, self.CXPB, self.MUTPB)

            # Evaluate the individuals with an invalid fitness (subject to crossover and mutation)
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = self.toolbox.map(self.toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            # The population is entirely replaced by the offspring
            self.pop[:] = offspring

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

        print(" -= END OF EVOLUTION =- ")
