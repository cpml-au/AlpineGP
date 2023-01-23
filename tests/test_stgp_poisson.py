import warnings
import numpy as np
from jax import grad
import jax.numpy as jnp
import jax.config as config
# import jaxopt
import math
import operator
# import jax

from deap import gp, tools
from dctkit.mesh import simplex, util
from dctkit.dec import cochain as C
from alpine.gp import gpsymbreg as gps
from jax import jit

import os
import gmsh
# import matplotlib.tri as tri
import matplotlib.pyplot as plt
import networkx as nx

import multiprocessing
# from memory_profiler import profile

from scipy.optimize import minimize

config.update("jax_enable_x64", True)

cwd = os.path.dirname(simplex.__file__)


def add(a, b):
    return a + b


def scalar_mul_float(a, b):
    return a*b


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

    u_true = jnp.array(node_coords[:, 0]**2 + node_coords[:, 1]**2)
    b_values = u_true[bnodes]

    '''
    triang = tri.Triangulation(node_coords[:, 0], node_coords[:, 1])
    plt.tricontourf(triang, u_true, cmap='RdBu', levels=20)
    plt.triplot(triang, 'ko-')
    plt.colorbar()
    plt.show()
    '''

    boundary_values = (jnp.array(bnodes), jnp.array(b_values))

    dim_0 = S.num_nodes
    f_vec = jnp.array(4.*np.ones(dim_0))
    f = C.CochainP0(S, f_vec, type="jax")

    u_0_vec = 0.01*np.random.rand(dim_0)
    u_0 = C.CochainP0(S, u_0_vec, type="jax")

    k = 1.0
    gamma = 1000.
    return u_0, u_true, S, f, k, boundary_values, gamma


u_0, u_true, S, f, k, boundary_values, gamma = generate_mesh_poisson("test3.msh")

# define primitive set
pset = gp.PrimitiveSetTyped("MAIN", [C.CochainP0], float,  "u")
# define cochain operations

# sum
pset.addPrimitive(add, [float, float], float, name="Add")
pset.addPrimitive(C.add, [C.CochainP0, C.CochainP0], C.CochainP0)
pset.addPrimitive(C.add, [C.CochainP1, C.CochainP1], C.CochainP1)


# coboundary
pset.addPrimitive(C.coboundary, [C.CochainP0], C.CochainP1, name="CoboundaryP0")
pset.addPrimitive(C.coboundary, [C.CochainP1], C.CochainP2, name="CoboundaryP1")
pset.addPrimitive(C.coboundary, [C.CochainD0], C.CochainD1, name="CoboundaryD0")
pset.addPrimitive(C.coboundary, [C.CochainD1], C.CochainD2, name="CoboundaryD1")

# hodge star
pset.addPrimitive(C.star, [C.CochainP0], C.CochainD2, name="Star0")
pset.addPrimitive(C.star, [C.CochainP1], C.CochainD1, name="Star1")
pset.addPrimitive(C.star, [C.CochainP2], C.CochainD0, name="Star2")

# scalar multiplication
pset.addPrimitive(C.scalar_mul, [C.CochainP0, float], C.CochainP0, "MulP0")
pset.addPrimitive(C.scalar_mul, [C.CochainP1, float], C.CochainP1, "MulP1")
pset.addPrimitive(C.scalar_mul, [C.CochainP2, float], C.CochainP2, "MulP2")
pset.addPrimitive(C.scalar_mul, [C.CochainD0, float], C.CochainD0, "MulD0")
pset.addPrimitive(C.scalar_mul, [C.CochainD1, float], C.CochainD1, "MulD1")
pset.addPrimitive(C.scalar_mul, [C.CochainD2, float], C.CochainD2, "MulD2")
pset.addPrimitive(scalar_mul_float, [float, float], float, "MulFloat")

# inner product
pset.addPrimitive(C.inner_product, [C.CochainP0, C.CochainP0], float, "Inner0")
pset.addPrimitive(C.inner_product, [C.CochainP1, C.CochainP1], float, "Inner1")
pset.addPrimitive(C.inner_product, [C.CochainP2, C.CochainP2], float, "Inner2")

# add sources and diffusivity
pset.addTerminal(f, C.CochainP0, name="f")
pset.addTerminal(k, float, name="k")
pset.addTerminal(0.5, float, name="1/2")


# extract positions and values of boundary nodes
pos, value = boundary_values


class ObjFunctional:
    def __init__(self) -> None:
        pass

    def setFunc(self, func, individual):
        self.energy_func = func
        self.individual = individual

    # @profile
    def evalEnergy(self, vec):
        # Transform the tree expression in a callable function
        penalty = 0.5*gamma*jnp.sum((vec[pos] - value)**2)

        c = C.CochainP0(S, vec, "jax")

        energy = self.energy_func(c) + penalty

        return energy


# suppress warnings
warnings.filterwarnings('ignore')

# define evaluation function

# @profile


def evalPoisson(individual):
    # NOTE: we are introducing a BIAS...
    if len(individual) > 15:
        return 100,

    energy_func = GPproblem.toolbox.compile(expr=individual)

    # the current objective function is the current energy
    obj = ObjFunctional()
    obj.setFunc(energy_func, individual)
    # minimize the energy w.r.t. data
    jac = jit(grad(obj.evalEnergy))
    # solver = jaxopt.LBFGS(obj.evalEnergy, maxiter=1000)
    # print("STOP")
    x = minimize(fun=obj.evalEnergy, x0=u_0.coeffs, method="BFGS", jac=jac).x

    result = np.linalg.norm(x-u_true)

    # to avoid strange numbers, if result is too distance from 0 or is nan we assign to
    # it a default big number
    if result > 10 or math.isnan(result):
        result = 100

    return result,


'''
limitHeight = 6
toolbox.decorate("mate", gp.staticLimit(
    key=operator.attrgetter("height"), max_value=limitHeight))
toolbox.decorate("mutate", gp.staticLimit(
    key=operator.attrgetter("height"), max_value=limitHeight))

limitLength = 15
toolbox.decorate("mate", gp.staticLimit(key=len, max_value=limitLength))
toolbox.decorate("mutate", gp.staticLimit(key=len, max_value=limitLength))
'''
NINDIVIDUALS = 500
NGEN = 20
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
    # pool = multiprocessing.Pool()
    # GPproblem.toolbox.register("map", pool.map)
    GPproblem.run(plot_history=True,
                  print_log=True,
                  plot_best=False,
                  seed=None)
    # pool.close()

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
