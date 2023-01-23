import warnings
import numpy as np
from jax import grad
import jax.numpy as jnp
import jax.config as config
# import jaxopt
import math
import operator

from deap import gp, tools
from dctkit.mesh import simplex, util
from dctkit.dec import cochain as C
from alpine.gp import gpsymbreg as gps
from alpine.data import poisson_dataset as data
from alpine.apps.poisson import pset
from jax import jit

import os
import gmsh
import matplotlib.pyplot as plt
import networkx as nx

import multiprocessing

from scipy.optimize import minimize

config.update("jax_enable_x64", True)

cwd = os.path.dirname(simplex.__file__)


def generate_mesh_poisson(filename):
    full_path = os.path.join(cwd, filename)
    _, _, S_2, node_coords = util.read_mesh(full_path)

    S = simplex.SimplicialComplex(S_2, node_coords)
    S.get_circumcenters()
    S.get_primal_volumes()
    S.get_dual_volumes()
    S.get_hodge_star()

    bnodes, _ = gmsh.model.mesh.getNodesForPhysicalGroup(1, 1)
    bnodes -= 1

    return S, bnodes


# generate mesh and dataset
S, bnodes = generate_mesh_poisson("test3.msh")
dim_0 = S.num_nodes
num_data = 2
data_X, data_y = data.generate_dataset(S, num_data)

bvalues = data_X[:, bnodes]
boundary_values = (jnp.array(bnodes), jnp.array(bvalues))

gamma = 1000.
u_0_vec = 0.01*np.random.rand(dim_0)
u_0 = C.CochainP0(S, u_0_vec, type="jax")

# FIXME: Fix validation process


class ObjFunctional:
    def __init__(self) -> None:
        pass

    def setFunc(self, func, individual):
        self.energy_func = func
        self.individual = individual

    # @profile
    def evalEnergy(self, vec_x, vec_y, vec_bvalues):
        # Transform the tree expression in a callable function
        penalty = 0.5*gamma*jnp.sum((vec_x[bnodes] - vec_bvalues)**2)
        # jax.debug.print("{x}", x=penalty)
        c = C.CochainP0(S, vec_x, "jax")
        fk = C.CochainP0(S, vec_y, "jax")
        # jax.debug.print("{x}", x=jnp.linalg.norm(c.coeffs - f.coeffs))
        energy = self.energy_func(c, fk) + penalty
        # jax.debug.print("{x}", x=energy)
        return energy


# suppress warnings
warnings.filterwarnings('ignore')


def evalPoisson(individual):
    # NOTE: we are introducing a BIAS...
    if (len(individual) < 10) or (len(individual) > 20):
        result = 1000
        return result,

    energy_func = GPproblem.toolbox.compile(expr=individual)

    # the current objective function is the current energy
    obj = ObjFunctional()
    obj.setFunc(energy_func, individual)
    result = 0
    for i, vec_y in enumerate(data_y):
        # minimize the energy w.r.t. data
        jac = jit(grad(obj.evalEnergy))

        # extract current bvalues
        vec_bvalues = bvalues[i, :]

        x = minimize(fun=obj.evalEnergy, x0=u_0.coeffs,
                     args=(vec_y, vec_bvalues), method="BFGS", jac=jac).x
        current_result = np.linalg.norm(x-data_X[i, :])**2

    # to avoid strange numbers, if result is too distance from 0 or is nan we assign to
    # it a default big number
    if current_result > 100 or math.isnan(current_result):
        current_result = 100

        result += current_result

    result = 1/(3*num_data)*result
    length_factor = math.prod([1 - i/len(individual)
                              for i in range(10, 21)])
    penalty_length = gamma*abs(length_factor)
    result += penalty_length
    return result,


NINDIVIDUALS = 10
NGEN = 1
CXPB = 0.5
MUTPB = 0.1

GPproblem = gps.GPSymbRegProblem(pset,
                                 NINDIVIDUALS,
                                 NGEN,
                                 CXPB,
                                 MUTPB,
                                 min_=1,
                                 max_=4)

# Register fitness function, selection and mutate operators
GPproblem.toolbox.register("evaluate", evalPoisson)
GPproblem.toolbox.register(
    "select", GPproblem.selElitistAndTournament, frac_elitist=0.1)
GPproblem.toolbox.register("mate", gp.cxOnePoint)
GPproblem.toolbox.register("expr_mut", gp.genGrow, min_=1, max_=3)
GPproblem.toolbox.register("mutate",
                           gp.mutUniform,
                           expr=GPproblem.toolbox.expr_mut,
                           pset=pset)

# Bloat control
GPproblem.toolbox.decorate(
    "mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
GPproblem.toolbox.decorate(
    "mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))


def test_stgp_poisson():

    # start learning
    pool = multiprocessing.Pool()
    GPproblem.toolbox.register("map", pool.map)
    GPproblem.run(plot_history=True,
                  print_log=True,
                  plot_best=False,
                  seed=None)
    pool.close()

    # Print best individual
    best = tools.selBest(GPproblem.pop, k=1)
    print(str(best[0]))

    # plot the best solution
    nodes, edges, labels = gp.graph(best[0])
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
