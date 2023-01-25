from deap import gp
import numpy as np
import warnings
import jax.numpy as jnp
from jax import jit, grad
from scipy.optimize import minimize
import operator
import math

from dctkit.dec import cochain as C
from alpine.data import poisson_dataset as d
from alpine.gp import gpsymbreg as gps


def add(a, b):
    return a + b


def scalar_mul_float(a, b):
    return a*b


# define primitive set
pset = gp.PrimitiveSetTyped("MAIN", [C.CochainP0, C.CochainP0], float,  "u")
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

# add constant = 0.5
pset.addTerminal(0.5, float, name="1/2")

# generate mesh and dataset
S, bnodes = d.generate_complex("test3.msh")
dim_0 = S.num_nodes
num_data = 1
diff = 1
k = 2
is_valid = False
X, y, kf = d.split_dataset(S, num_data, diff, k)

if is_valid:
    X_train, X_test = X
    y_train, y_test = y
    # extract bvalues_test
    bvalues_test = X_test[:, bnodes]
    dataset = (X_train, y_train, X_test, y_test)
else:
    dataset = (X, y)
    X_train = X
    y_train = y
    # otherwise I have problem with import
    bvalues_test = None

# extract bvalues_train
bvalues_train = X_train[:, bnodes]

gamma = 1000.
u_0_vec = 0.01*np.random.rand(dim_0)
u_0 = C.CochainP0(S, u_0_vec, type="jax")


class ObjFunctional:
    def __init__(self) -> None:
        pass

    def setFunc(self, func, individual):
        self.energy_func = func
        self.individual = individual

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


def evalPoisson(individual, X, y, current_bvalues):
    # NOTE: we are introducing a BIAS...
    if len(individual) > 50:
        result = 1000
        # print(result)
        return result,

    energy_func = GPproblem.toolbox.compile(expr=individual)

    # the current objective function is the current energy
    obj = ObjFunctional()
    obj.setFunc(energy_func, individual)

    result = 0
    for i, vec_y in enumerate(y):
        # minimize the energy w.r.t. data
        jac = jit(grad(obj.evalEnergy))

        # extract current bvalues
        vec_bvalues = current_bvalues[i, :]

        x = minimize(fun=obj.evalEnergy, x0=u_0.coeffs,
                     args=(vec_y, vec_bvalues), method="BFGS", jac=jac).x
        current_result = np.linalg.norm(x-X[i, :])**2

        # to avoid strange numbers, if result is too distance from 0 or is nan we
        # assign to it a default big number
        if current_result > 100 or math.isnan(current_result):
            current_result = 100

        result += current_result

    result = 1/(diff*num_data)*result
    # length_factor = math.prod([1 - i/len(individual)
    # for i in range(0, 50)])
    # penalty_length = gamma*abs(length_factor)
    # result += penalty_length
    return result,


NINDIVIDUALS = 350
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


FinalGP = GPproblem

# Set toolbox for FinalGP
FinalGP.toolbox.register("evaluate", evalPoisson, X=X_train,
                         y=y_train, current_bvalues=bvalues_train)
