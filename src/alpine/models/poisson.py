from deap import gp
from dctkit.dec import cochain as C
import math
import operator

# FIXME: these functions are not specific to Poisson, move elsewhere


def protectedDiv(left, right):
    try:
        return left / right
    except ZeroDivisionError:
        return math.nan


# define primitive set
pset = gp.PrimitiveSetTyped("MAIN", [C.CochainP0, C.CochainP0], float)

# scalar operations
pset.addPrimitive(operator.add, [float, float], float, name="Add")
pset.addPrimitive(operator.sub, [float, float], float, name="Sub")
pset.addPrimitive(operator.mul, [float, float], float, name="MulF")
pset.addPrimitive(protectedDiv, [float, float], float, name="Div")

# cochain operations
pset.addPrimitive(C.add, [C.CochainP0, C.CochainP0], C.CochainP0, name="AddP0")
pset.addPrimitive(C.add, [C.CochainP1, C.CochainP1], C.CochainP1, name="AddP1")
pset.addPrimitive(C.sub, [C.CochainP0, C.CochainP0], C.CochainP0, name="SubP0")
# pset.addPrimitive(C.sub, [C.CochainP1, C.CochainP1], C.CochainP1, name="SubP1")

pset.addPrimitive(C.coboundary, [C.CochainP0], C.CochainP1, name="dP0")
pset.addPrimitive(C.coboundary, [C.CochainP1], C.CochainP2, name="dP1")
pset.addPrimitive(C.coboundary, [C.CochainD0], C.CochainD1, name="dD0")
pset.addPrimitive(C.coboundary, [C.CochainD1], C.CochainD2, name="dD1")
pset.addPrimitive(C.codifferential, [C.CochainP1], C.CochainP0, name="delP1")
pset.addPrimitive(C.codifferential, [C.CochainP2], C.CochainP1, name="delP2")

pset.addPrimitive(C.star, [C.CochainP0], C.CochainD2, name="St0")
pset.addPrimitive(C.star, [C.CochainP1], C.CochainD1, name="St1")
pset.addPrimitive(C.star, [C.CochainP2], C.CochainD0, name="St2")

pset.addPrimitive(C.scalar_mul, [C.CochainP0, float], C.CochainP0, "MulP0")
pset.addPrimitive(C.scalar_mul, [C.CochainP1, float], C.CochainP1, "MulP1")
pset.addPrimitive(C.scalar_mul, [C.CochainP2, float], C.CochainP2, "MulP2")
pset.addPrimitive(C.scalar_mul, [C.CochainD0, float], C.CochainD0, "MulD0")
pset.addPrimitive(C.scalar_mul, [C.CochainD1, float], C.CochainD1, "MulD1")
pset.addPrimitive(C.scalar_mul, [C.CochainD2, float], C.CochainD2, "MulD2")

pset.addPrimitive(C.inner_product, [C.CochainP0, C.CochainP0], float, "Inn0")
pset.addPrimitive(C.inner_product, [C.CochainP1, C.CochainP1], float, "Inn1")
pset.addPrimitive(C.inner_product, [C.CochainP2, C.CochainP2], float, "Inn2")

# add constant = 0.5
pset.addTerminal(0.5, float, name="1/2")

# rename arguments
pset.renameArguments(ARG0="u")
pset.renameArguments(ARG1="fk")
