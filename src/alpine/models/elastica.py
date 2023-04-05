from deap import gp
from dctkit.dec import cochain as C
import math
import jax.numpy as jnp
import operator
import dctkit as dt

# FIXME: these functions are not specific to elastica, move elsewhere


def protectedDiv(left, right):
    try:
        return jnp.array(left / right, dtype=dt.float_dtype)
    except ZeroDivisionError:
        return jnp.nan


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


def add_mod(x, y):
    return jnp.array(operator.add(x, y), dtype=dt.float_dtype)


def sub_mod(x, y):
    return jnp.array(operator.sub(x, y), dtype=dt.float_dtype)


def mul_mod(x, y):
    return jnp.array(operator.mul(x, y), dtype=dt.float_dtype)


def square_mod(x):
    return jnp.square(x).astype(dt.float_dtype)


# define primitive set
pset = gp.PrimitiveSetTyped("MAIN", [C.CochainD0, float], float)

# scalar operations
pset.addPrimitive(add_mod, [float, float], float, name="Add")
pset.addPrimitive(sub_mod, [float, float], float, name="Sub")
pset.addPrimitive(mul_mod, [float, float], float, name="MulF")
pset.addPrimitive(protectedDiv, [float, float], float, name="Div")
pset.addPrimitive(jnp.sin, [float], float, name="SinF")
pset.addPrimitive(jnp.arcsin, [float], float, name="ArcsinF")
pset.addPrimitive(jnp.cos, [float], float, name="CosF")
pset.addPrimitive(jnp.arccos, [float], float, name="ArccosF")
pset.addPrimitive(jnp.exp, [float], float, name="ExpF")
pset.addPrimitive(protectedLog, [float], float, name="LogF")
pset.addPrimitive(protectedSqrt, [float], float, name="SqrtF")
pset.addPrimitive(square_mod, [float], float, name="SquareF")

# cochain operations
pset.addPrimitive(C.add, [C.CochainP0, C.CochainP0], C.CochainP0, name="AddP0")
pset.addPrimitive(C.add, [C.CochainP1, C.CochainP1], C.CochainP1, name="AddP1")
pset.addPrimitive(C.sub, [C.CochainP0, C.CochainP0], C.CochainP0, name="SubP0")
pset.addPrimitive(C.sub, [C.CochainP1, C.CochainP1], C.CochainP1, name="SubP1")

pset.addPrimitive(C.coboundary, [C.CochainP0], C.CochainP1, name="dP0")
pset.addPrimitive(C.coboundary, [C.CochainD0], C.CochainD1, name="dD0")
pset.addPrimitive(C.codifferential, [C.CochainP1], C.CochainP0, name="delP1")
pset.addPrimitive(C.codifferential, [C.CochainD1], C.CochainD0, name="delD1")

pset.addPrimitive(C.star, [C.CochainP0], C.CochainD1, name="St0")
pset.addPrimitive(C.star, [C.CochainP1], C.CochainD0, name="St1")
pset.addPrimitive(C.star, [C.CochainD0], C.CochainP1, name="InvSt0")
pset.addPrimitive(C.star, [C.CochainP1], C.CochainD0, name="InvSt1")

pset.addPrimitive(C.scalar_mul, [C.CochainP0, float], C.CochainP0, "MulP0")
pset.addPrimitive(C.scalar_mul, [C.CochainP1, float], C.CochainP1, "MulP1")
pset.addPrimitive(C.scalar_mul, [C.CochainD0, float], C.CochainD0, "MulD0")
pset.addPrimitive(C.scalar_mul, [C.CochainD1, float], C.CochainD1, "MulD1")

pset.addPrimitive(C.cochain_mul, [C.CochainP0, C.CochainP0], C.CochainP0, "CochMulP0")
pset.addPrimitive(C.cochain_mul, [C.CochainP1, C.CochainP1], C.CochainP1, "CochMulP1")
pset.addPrimitive(C.cochain_mul, [C.CochainD0, C.CochainD0], C.CochainD0, "CochMulD0")
pset.addPrimitive(C.cochain_mul, [C.CochainD1, C.CochainD1], C.CochainD1, "CochMulD1")

pset.addPrimitive(C.inner_product, [C.CochainP0, C.CochainP0], float, "InnP0")
pset.addPrimitive(C.inner_product, [C.CochainP1, C.CochainP1], float, "InnP1")
pset.addPrimitive(C.inner_product, [C.CochainD0, C.CochainD0], float, "InnD0")
pset.addPrimitive(C.inner_product, [C.CochainD1, C.CochainD1], float, "InnD1")

pset.addPrimitive(C.sin, [C.CochainP0], C.CochainP0, "SinP0")
pset.addPrimitive(C.sin, [C.CochainP1], C.CochainP1, "SinP1")
pset.addPrimitive(C.sin, [C.CochainD0], C.CochainD0, "SinD0")
pset.addPrimitive(C.sin, [C.CochainD1], C.CochainD1, "SinD1")

# add constants
pset.addTerminal(0.5, float, name="1/2")
pset.addTerminal(-1., float, name="-1")
pset.addTerminal(2., float, name="2")

# rename arguments
pset.renameArguments(ARG0="theta_coch")
pset.renameArguments(ARG1="FL2_EI_0")
