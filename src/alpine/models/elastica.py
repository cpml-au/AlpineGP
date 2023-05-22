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


def add(x, y):
    return jnp.add(x, y).astype(dt.float_dtype)


def sub(x, y):
    return jnp.subtract(x, y).astype(dt.float_dtype)


def mul(x, y):
    return jnp.multiply(x, y).astype(dt.float_dtype)


def square_mod(x):
    return jnp.square(x).astype(dt.float_dtype)


def inv_float(x):
    try:
        return 1/x
    except ZeroDivisionError:
        return jnp.nan


def inv_scalar_mul(c, f):
    try:
        return C.scalar_mul(c, 1/f)
    except ZeroDivisionError:
        return C.scalar_mul(c, jnp.nan)



def add_primitives(pset: gp.PrimitiveSetTyped) -> None:
    # scalar operations
    pset.addPrimitive(add, [float, float], float, name="Add")
    pset.addPrimitive(sub, [float, float], float, name="Sub")
    pset.addPrimitive(mul, [float, float], float, name="MulF")
    pset.addPrimitive(protectedDiv, [float, float], float, name="Div")
    pset.addPrimitive(jnp.sin, [float], float, name="SinF")
    pset.addPrimitive(jnp.arcsin, [float], float, name="ArcsinF")
    pset.addPrimitive(jnp.cos, [float], float, name="CosF")
    pset.addPrimitive(jnp.arccos, [float], float, name="ArccosF")
    pset.addPrimitive(jnp.exp, [float], float, name="ExpF")
    pset.addPrimitive(protectedLog, [float], float, name="LogF")
    pset.addPrimitive(protectedSqrt, [float], float, name="SqrtF")
    pset.addPrimitive(square_mod, [float], float, name="SquareF")
    pset.addPrimitive(inv_float, [float], float, name="InvF")

    # cochain operations
    pset.addPrimitive(C.add, [C.CochainP0, C.CochainP0], C.CochainP0, name="AddP0")
    pset.addPrimitive(C.add, [C.CochainP1, C.CochainP1], C.CochainP1, name="AddP1")
    pset.addPrimitive(C.add, [C.CochainD0, C.CochainD0], C.CochainD0, name="AddD0")
    pset.addPrimitive(C.add, [C.CochainD1, C.CochainD1], C.CochainD1, name="AddD1")
    pset.addPrimitive(C.sub, [C.CochainP0, C.CochainP0], C.CochainP0, name="SubP0")
    pset.addPrimitive(C.sub, [C.CochainP1, C.CochainP1], C.CochainP1, name="SubP1")
    pset.addPrimitive(C.sub, [C.CochainD0, C.CochainD0], C.CochainD0, name="SubD0")
    pset.addPrimitive(C.sub, [C.CochainD1, C.CochainD1], C.CochainD1, name="SubD1")

    pset.addPrimitive(C.coboundary, [C.CochainP0], C.CochainP1, name="dP0")
    pset.addPrimitive(C.coboundary, [C.CochainD0], C.CochainD1, name="dD0")
    pset.addPrimitive(C.codifferential, [C.CochainP1], C.CochainP0, name="delP1")
    pset.addPrimitive(C.codifferential, [C.CochainD1], C.CochainD0, name="delD1")

    pset.addPrimitive(C.star, [C.CochainP0], C.CochainD1, name="St0")
    pset.addPrimitive(C.star, [C.CochainP1], C.CochainD0, name="St1")
    pset.addPrimitive(C.star, [C.CochainD1], C.CochainP0, name="InvSt0")
    pset.addPrimitive(C.star, [C.CochainD0], C.CochainP1, name="InvSt1")

    pset.addPrimitive(C.scalar_mul, [C.CochainP0, float], C.CochainP0, "MulP0")
    pset.addPrimitive(C.scalar_mul, [C.CochainP1, float], C.CochainP1, "MulP1")
    pset.addPrimitive(C.scalar_mul, [C.CochainD0, float], C.CochainD0, "MulD0")
    pset.addPrimitive(C.scalar_mul, [C.CochainD1, float], C.CochainD1, "MulD1")
    pset.addPrimitive(inv_scalar_mul, [C.CochainP0, float], C.CochainP0, "InvMulP0")
    pset.addPrimitive(inv_scalar_mul, [C.CochainP1, float], C.CochainP1, "InvMulP1")
    pset.addPrimitive(inv_scalar_mul, [C.CochainD0, float], C.CochainD0, "InvMulD0")
    pset.addPrimitive(inv_scalar_mul, [C.CochainD1, float], C.CochainD1, "InvMulD1")


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

    pset.addPrimitive(C.arcsin, [C.CochainP0], C.CochainP0, "ArcsinP0")
    pset.addPrimitive(C.arcsin, [C.CochainP1], C.CochainP1, "ArcsinP1")
    pset.addPrimitive(C.arcsin, [C.CochainD0], C.CochainD0, "ArcsinD0")
    pset.addPrimitive(C.arcsin, [C.CochainD1], C.CochainD1, "ArcsinD1")

    pset.addPrimitive(C.cos, [C.CochainP0], C.CochainP0, "CosP0")
    pset.addPrimitive(C.cos, [C.CochainP1], C.CochainP1, "CosP1")
    pset.addPrimitive(C.cos, [C.CochainD0], C.CochainD0, "CosD0")
    pset.addPrimitive(C.cos, [C.CochainD1], C.CochainD1, "CosD1")

    pset.addPrimitive(C.arccos, [C.CochainP0], C.CochainP0, "ArccosP0")
    pset.addPrimitive(C.arccos, [C.CochainP1], C.CochainP1, "ArccosP1")
    pset.addPrimitive(C.arccos, [C.CochainD0], C.CochainD0, "ArccosD0")
    pset.addPrimitive(C.arccos, [C.CochainD1], C.CochainD1, "ArccosD1")

    pset.addPrimitive(C.exp, [C.CochainP0], C.CochainP0, "ExpP0")
    pset.addPrimitive(C.exp, [C.CochainP1], C.CochainP1, "ExpP1")
    pset.addPrimitive(C.exp, [C.CochainD0], C.CochainD0, "ExpD0")
    pset.addPrimitive(C.exp, [C.CochainD1], C.CochainD1, "ExpD1")

    pset.addPrimitive(C.log, [C.CochainP0], C.CochainP0, "LogP0")
    pset.addPrimitive(C.log, [C.CochainP1], C.CochainP1, "LogP1")
    pset.addPrimitive(C.log, [C.CochainD0], C.CochainD0, "LogD0")
    pset.addPrimitive(C.log, [C.CochainD1], C.CochainD1, "LogD1")

    pset.addPrimitive(C.sqrt, [C.CochainP0], C.CochainP0, "SqrtP0")
    pset.addPrimitive(C.sqrt, [C.CochainP1], C.CochainP1, "SqrtP1")
    pset.addPrimitive(C.sqrt, [C.CochainD0], C.CochainD0, "SqrtD0")
    pset.addPrimitive(C.sqrt, [C.CochainD1], C.CochainD1, "SqrtD1")

    pset.addPrimitive(C.square, [C.CochainP0], C.CochainP0, "SquareP0")
    pset.addPrimitive(C.square, [C.CochainP1], C.CochainP1, "SquareP1")
    pset.addPrimitive(C.square, [C.CochainD0], C.CochainD0, "SquareD0")
    pset.addPrimitive(C.square, [C.CochainD1], C.CochainD1, "SquareD1")

    # add constants
    pset.addTerminal(0.5, float, name="1/2")
    pset.addTerminal(-1., float, name="-1")
    pset.addTerminal(2., float, name="2")

    # rename arguments
    pset.renameArguments(ARG0="theta")
    pset.renameArguments(ARG1="FL2_EI0")
