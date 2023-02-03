from deap import gp
from dctkit.dec import cochain as C
import math

# FIXME: these functions are not specific to Poisson, move elsewhere


def add(a, b):
    return a + b


def scalar_mul_float(a, b):
    return a*b


def protectedDiv(left, right):
    try:
        return left / right
    except ZeroDivisionError:
        return math.nan


# define primitive set
pset = gp.PrimitiveSetTyped("MAIN", [C.CochainP0, C.CochainP0], float)
# define cochain operations


# sum
pset.addPrimitive(add, [float, float], float, name="Add")
pset.addPrimitive(C.add, [C.CochainP0, C.CochainP0], C.CochainP0, name="AddP0")
pset.addPrimitive(C.add, [C.CochainP1, C.CochainP1], C.CochainP1, name="AddP1")


# coboundary
pset.addPrimitive(C.coboundary, [C.CochainP0], C.CochainP1, name="CoboundaryP0")
pset.addPrimitive(C.coboundary, [C.CochainP1], C.CochainP2, name="CoboundaryP1")
pset.addPrimitive(C.coboundary, [C.CochainD0], C.CochainD1, name="CoboundaryD0")
pset.addPrimitive(C.coboundary, [C.CochainD1], C.CochainD2, name="CoboundaryD1")

# hodge star
pset.addPrimitive(C.star, [C.CochainP0], C.CochainD2, name="Star0")
pset.addPrimitive(C.star, [C.CochainP1], C.CochainD1, name="Star1")
pset.addPrimitive(C.star, [C.CochainP2], C.CochainD0, name="Star2")

# scalar multiplication/division
pset.addPrimitive(C.scalar_mul, [C.CochainP0, float], C.CochainP0, "MulP0")
pset.addPrimitive(C.scalar_mul, [C.CochainP1, float], C.CochainP1, "MulP1")
pset.addPrimitive(C.scalar_mul, [C.CochainP2, float], C.CochainP2, "MulP2")
pset.addPrimitive(C.scalar_mul, [C.CochainD0, float], C.CochainD0, "MulD0")
pset.addPrimitive(C.scalar_mul, [C.CochainD1, float], C.CochainD1, "MulD1")
pset.addPrimitive(C.scalar_mul, [C.CochainD2, float], C.CochainD2, "MulD2")
pset.addPrimitive(scalar_mul_float, [float, float], float, "MulFloat")
pset.addPrimitive(protectedDiv, [float, float], float, name="Div")

# inner product
pset.addPrimitive(C.inner_product, [C.CochainP0, C.CochainP0], float, "Inner0")
pset.addPrimitive(C.inner_product, [C.CochainP1, C.CochainP1], float, "Inner1")
pset.addPrimitive(C.inner_product, [C.CochainP2, C.CochainP2], float, "Inner2")

# add constant = 0.5
pset.addTerminal(0.5, float, name="1/2")

# rename arguments
pset.renameArguments(ARG0="u")
pset.renameArguments(ARG1="fk")
