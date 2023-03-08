import numpy as np
import jax.numpy as jnp

from dctkit.mesh import simplex, util
from dctkit.dec import cochain as C
from deap import base, algorithms, tools, gp, creator


import os
import gmsh
import matplotlib.tri as tri
import matplotlib.pyplot as plt
import networkx as nx

import multiprocessing

cwd = os.path.dirname(simplex.__file__)


def generate_mesh(filename):
    full_path = os.path.join(cwd, filename)
    _, _, S_2, node_coords = util.read_mesh(full_path)

    S = simplex.SimplicialComplex(S_2, node_coords)
    S.get_circumcenters()
    S.get_primal_volumes()
    S.get_dual_volumes()
    S.get_hodge_star()

    bnodes, _ = gmsh.model.mesh.getNodesForPhysicalGroup(1, 1)
    bnodes -= 1
    triang = tri.Triangulation(node_coords[:, 0], node_coords[:, 1])

    u_true = jnp.array(node_coords[:, 0]**2 + node_coords[:, 1]**2)
    b_values = u_true[bnodes]

    plt.tricontourf(triang, u_true, cmap='RdBu', levels=20)
    plt.triplot(triang, 'ko-')
    plt.colorbar()
    plt.show()

    boundary_values = (jnp.array(bnodes), jnp.array(b_values))

    dim_0 = S.num_nodes
    f_vec = jnp.array(4.*np.ones(dim_0))
    f = C.CochainP0(S, f_vec)

    u_0_vec = 0.01*np.random.rand(dim_0)
    u_0 = C.CochainP0(S, u_0_vec)

    k = 1.0/2
    gamma = 100.
    return u_0, u_true, S, f, k, boundary_values, gamma


u_0, u_true, S, f, k, boundary_values, gamma = generate_mesh("test1.msh")
# initialize dataset
data_input = []
data_output = []

# set seed
np.random.seed(42)

# fill dataset
for _ in range(100):
    in_c = C.Cochain(dim=0, is_primal=True, complex=S,
                     coeffs=np.random.randint(10, size=5))
    out_c = C.star(C.coboundary(in_c))
    data_input.append(in_c)
    data_output.append(out_c)

# define primitive set
pset = gp.PrimitiveSetTyped("MAIN", [C.CochainP0], C.CochainD1, "u")

# define cochain operations
pset.addPrimitive(C.add, [C.CochainP0, C.CochainP0], C.CochainP0)
pset.addPrimitive(C.add, [C.CochainP1, C.CochainP1], C.CochainP1)
pset.addPrimitive(C.coboundary, [C.CochainP0], C.CochainP1, name="CoboundaryP0")
pset.addPrimitive(C.coboundary, [C.CochainP1], C.CochainP2, name="CoboundaryP1")
pset.addPrimitive(C.coboundary, [C.CochainD0], C.CochainD1, name="CoboundaryD0")
pset.addPrimitive(C.coboundary, [C.CochainD1], C.CochainD2, name="CoboundaryD1")
pset.addPrimitive(C.star, [C.CochainP0], C.CochainD2, name="Star0")
pset.addPrimitive(C.star, [C.CochainP1], C.CochainD1, name="Star1")
pset.addPrimitive(C.star, [C.CochainP2], C.CochainD0, name="Star2")

# introduce identity primitives to prevent add primitive error
# pset.addPrimitive(C.identity, [C.CochainP0], C.CochainP0, name="Identity0")

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

toolbox = base.Toolbox()

# register how to create an individual
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=0, max_=1)

toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)

# define evaluation function


def evalToy(individual):
    # Transform the tree expression in a callable function
    func = toolbox.compile(expr=individual)
    # Evaluate the error

    # NOTE: change it if the dataset is not a list
    result = 1/len(data_input) * sum([((func(c).coeffs - d.coeffs)**2).mean()
                                      for c, d in zip(data_input, data_output)])
    return result,


# register the evaluation function in the toolbox
toolbox.register("evaluate", evalToy)

# register the selection method chosen
toolbox.register("select", tools.selTournament, tournsize=3)

# register the mate method
toolbox.register("mate", gp.cxOnePoint)

# register the mutation method
toolbox.register("expr_mut", gp.genFull, min_=1, max_=2)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)


def test_stgp_toy_example():
    pop = toolbox.population(n=10)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    # start learning
    pool = multiprocessing.Pool()
    toolbox.register("map", pool.map)
    algorithms.eaSimple(pop, toolbox, 0.5, 0.2, 5, stats, halloffame=hof)
    pool.close()
    print(hof[0])

    # plot the best solution
    nodes, edges, labels = gp.graph(hof[0])
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
    test_stgp_toy_example()
