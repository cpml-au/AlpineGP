from deap import gp
from dctkit.dec import cochain as C
import math
import jax.numpy as jnp
import operator

# FIXME: these functions are not specific to elastica, move elsewhere


def protectedDiv(left, right):
    try:
        return left / right
    except ZeroDivisionError:
        return math.nan


def protectedLog(x):
    try:
        return jnp.log(x)
    except ValueError:
        return jnp.nan


def protectedSqrt(x):
    try:
        return jnp.sqrt(x)
    except ValueError:
        return jnp.nan


# define primitive set
pset = gp.PrimitiveSetTyped("MAIN", [C.CochainD0, float], float)

# scalar operations
pset.addPrimitive(operator.add, [float, float], float, name="Add")
pset.addPrimitive(operator.sub, [float, float], float, name="Sub")
pset.addPrimitive(operator.mul, [float, float], float, name="MulF")
pset.addPrimitive(protectedDiv, [float, float], float, name="Div")
pset.addPrimitive(jnp.sin, [float], float, name="SinF")
pset.addPrimitive(jnp.arcsin, [float], float, name="ArcsinF")
pset.addPrimitive(jnp.cos, [float], float, name="CosF")
pset.addPrimitive(jnp.arccos, [float], float, name="ArccosF")
pset.addPrimitive(jnp.exp, [float], float, name="ExpF")
pset.addPrimitive(protectedLog, [float], float, name="LogF")
pset.addPrimitive(protectedSqrt, [float], float, name="SqrtF")
pset.addPrimitive(jnp.square, [float], float, name="SquareF")

# cochain operations
pset.addPrimitive(C.add, [C.CochainP0, C.CochainP0], C.CochainP0, name="AddP0")
pset.addPrimitive(C.add, [C.CochainP1, C.CochainP1], C.CochainP1, name="AddP1")
pset.addPrimitive(C.add, [C.CochainP2, C.CochainP2], C.CochainP2, name="AddP2")
pset.addPrimitive(C.sub, [C.CochainP0, C.CochainP0], C.CochainP0, name="SubP0")
pset.addPrimitive(C.sub, [C.CochainP1, C.CochainP1], C.CochainP1, name="SubP1")
pset.addPrimitive(C.sub, [C.CochainP2, C.CochainP2], C.CochainP2, name="SubP2")

pset.addPrimitive(C.coboundary, [C.CochainP0], C.CochainP1, name="dP0")
pset.addPrimitive(C.coboundary, [C.CochainP1], C.CochainP2, name="dP1")
pset.addPrimitive(C.coboundary, [C.CochainD0], C.CochainD1, name="dD0")
pset.addPrimitive(C.coboundary, [C.CochainD1], C.CochainD2, name="dD1")
pset.addPrimitive(C.codifferential, [C.CochainP1], C.CochainP0, name="delP1")
pset.addPrimitive(C.codifferential, [C.CochainP2], C.CochainP1, name="delP2")
pset.addPrimitive(C.codifferential, [C.CochainD1], C.CochainD0, name="delD1")
pset.addPrimitive(C.codifferential, [C.CochainD2], C.CochainD1, name="delD2")

pset.addPrimitive(C.star, [C.CochainP0], C.CochainD2, name="St0")
pset.addPrimitive(C.star, [C.CochainP1], C.CochainD1, name="St1")
pset.addPrimitive(C.star, [C.CochainP2], C.CochainD0, name="St2")
pset.addPrimitive(C.star, [C.CochainD0], C.CochainP2, name="InvSt0")
pset.addPrimitive(C.star, [C.CochainD1], C.CochainP1, name="InvSt1")
pset.addPrimitive(C.star, [C.CochainD2], C.CochainP0, name="InvSt2")

pset.addPrimitive(C.scalar_mul, [C.CochainP0, float], C.CochainP0, "MulP0")
pset.addPrimitive(C.scalar_mul, [C.CochainP1, float], C.CochainP1, "MulP1")
pset.addPrimitive(C.scalar_mul, [C.CochainP2, float], C.CochainP2, "MulP2")
pset.addPrimitive(C.scalar_mul, [C.CochainD0, float], C.CochainD0, "MulD0")
pset.addPrimitive(C.scalar_mul, [C.CochainD1, float], C.CochainD1, "MulD1")
pset.addPrimitive(C.scalar_mul, [C.CochainD2, float], C.CochainD2, "MulD2")

pset.addPrimitive(C.inner_product, [C.CochainP0, C.CochainP0], float, "InnP0")
pset.addPrimitive(C.inner_product, [C.CochainP1, C.CochainP1], float, "InnP1")
pset.addPrimitive(C.inner_product, [C.CochainP2, C.CochainP2], float, "InnP2")
pset.addPrimitive(C.inner_product, [C.CochainD0, C.CochainD0], float, "InnD0")
pset.addPrimitive(C.inner_product, [C.CochainD1, C.CochainD1], float, "InnD1")
pset.addPrimitive(C.inner_product, [C.CochainD2, C.CochainD2], float, "InnD2")

pset.addPrimitive(C.sin, [C.CochainP0], C.CochainP0, "SinP0")
pset.addPrimitive(C.sin, [C.CochainP1], C.CochainP1, "SinP1")
pset.addPrimitive(C.sin, [C.CochainP2], C.CochainP2, "SinP2")
pset.addPrimitive(C.sin, [C.CochainD0], C.CochainD0, "SinD0")
pset.addPrimitive(C.sin, [C.CochainD1], C.CochainD1, "SinD1")
pset.addPrimitive(C.sin, [C.CochainD2], C.CochainD2, "SinD2")

# add constants
pset.addTerminal(0.5, float, name="1/2")
pset.addTerminal(-1., float, name="-1")
pset.addTerminal(2., float, name="2")

# rename arguments
pset.renameArguments(ARG0="theta")
pset.renameArguments(ARG1="EI0")
