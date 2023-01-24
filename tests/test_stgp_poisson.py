import multiprocessing
import networkx as nx
import matplotlib.pyplot as plt
import jax.config as config
from deap import gp, tools
from alpine.apps.poisson import bnodes, bvalues_test, kf, dataset, GPproblem, FinalGP, evalPoisson


config.update("jax_enable_x64", True)


def test_stgp_poisson():
    # initialize list of best individuals and list of best scores
    best_individuals = []
    best_train_scores = []
    best_val_scores = []

    # extract dataset
    X_train, y_train, X_test, y_test = dataset

    # start learning
    for train_index, valid_index in kf.split(X_train, y_train):
        # divide the dataset in training and validation set
        X_t, X_val = X_train[train_index, :], X_train[valid_index, :]
        y_t, y_val = y_train[train_index, :], y_train[valid_index, :]

        # define current bvalues datasets
        current_bvalues_train = X_t[:, bnodes]
        current_bvalues_val = X_val[:, bnodes]

        # update toolbox
        GPproblem.toolbox.register("evaluate", evalPoisson,
                                   X=X_t, y=y_t, current_bvalues=current_bvalues_train)

        # train the model in the training set
        pool = multiprocessing.Pool()
        GPproblem.toolbox.register("map", pool.map)
        GPproblem.run(plot_history=True,
                      print_log=True,
                      plot_best=True,
                      seed=None)
        pool.close()

        # Print best individual
        best = tools.selBest(GPproblem.pop, k=1)
        print(f"The best individual in this fold is {str(best[0])}")

        # evaluate score on the current training and validation set
        score_train = GPproblem.min_history[-1]
        score_val = evalPoisson(best[0], X_val, y_val, current_bvalues_val)
        score_val = score_val[0]

        print(f"The best score on training set in this fold is {score_train}")
        print(f"The best score on validation set in this fold is {score_val}")

        # save best individual and best score on training and validation set
        best_individuals.append(best[0])

        # FIXME: do I need it?
        best_train_scores.append(score_train)
        best_val_scores.append(score_train)

        print("-FOLD COMPLETED-")

    # first individuals of the final population are the best individuals so far
    for i in range(len(best_individuals)):
        FinalGP.pop[i] = best_individuals[i]

    # now we retrain all the k best models on the entire training set
    pool = multiprocessing.Pool()
    FinalGP.toolbox.register("map", pool.map)
    FinalGP.run(plot_history=True,
                print_log=True,
                plot_best=True,
                seed=None)
    pool.close()
    real_best = tools.selBest(FinalGP.pop, k=1)

    score_train = FinalGP.min_history[-1]
    score_test = evalPoisson(real_best[0], X_test, y_test, bvalues_test)
    score_test = score_test[0]

    print(f"The best score on training set in this fold is {score_train}")
    print(f"The best score on validation set in this fold is {score_test}")

    # plot the best solution
    nodes, edges, labels = gp.graph(real_best[0])
    graph = nx.Graph()
    graph.add_nodes_from(nodes)
    graph.add_edges_from(edges)
    # pos = graphviz_layout(graph, prog='dot')
    pos = nx.nx_agraph.graphviz_layout(graph, prog="dot")

    plt.figure(figsize=(7, 7))
    nx.draw_networkx_nodes(graph, pos, node_size=900, node_color="w")
    nx.draw_networkx_edges(graph, pos)
    nx.draw_networkx_labels(graph, pos, labels)
    plt.axis("off")
    plt.show()


if __name__ == '__main__':
    test_stgp_poisson()
