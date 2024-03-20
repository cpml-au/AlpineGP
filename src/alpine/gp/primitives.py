import jax.numpy as jnp
import numpy as np
from deap import gp
from dctkit.dec import cochain as C
import dctkit as dt
from typing import List, Dict, Callable, Tuple
from functools import partial
import operator


def protectedDiv(left, right):
    try:
        return jnp.divide(left, right)
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


def inv_float(x):
    return protectedDiv(1., x)


def inv_scalar_mul(c, f):
    try:
        return C.scalar_mul(c, 1/f)
    except ZeroDivisionError:
        return C.scalar_mul(c, jnp.nan)


def square_mod(x):
    return jnp.square(x).astype(dt.float_dtype)


def switch_category(categories: Tuple, category: str):
    switched_category_list = list(set(categories) - set(category))
    return str(switched_category_list[0])


class PrimitiveParams:
    def __init__(self, op, in_types, out_type) -> None:
        self.op = op
        self.in_types = in_types
        self.out_type = out_type


def generate_primitive(primitive: Dict[str, Dict[str, Callable] | List[str] | str |
                                       Dict]) -> Dict:
    """Generate all the primitives given a typed function.

    Args:
        primitive: a dictionary containing the relevant information of the function.
          It consists of the following 5 keys: 'fun_info' contains an inner dictionary
          encoding the name of the function (value of the inner key 'name') and the
          callable itself (value of the inner key 'fun'); 'input' contains a list
          composed of the input types; 'output' contains a string encoding the output
          type; 'att_input' contains an inner dictionary with keys 'category'
          (primal/dual), 'dimension' (0,1,2) and 'rank' ("SC", i.e. scalar, "V", "T"
          or "VT"); 'map_output' contains an inner dictionary consisting of the
          same keys of 'att_input'. In this case, each key contains a callable object
          that provides the map to get the output category/dimension/rank given the
          input one.

    Returns:
        a dict in which each key is the name of the sub-primitive and each value
            is a PrimitiveParams object.
    """
    general_primitive = primitive['fun_info']
    in_attribute = primitive['att_input']
    map_rule = primitive['map_rule']
    primitive_dictionary = dict()
    for in_category in in_attribute['category']:
        for in_dim in in_attribute['dimension']:
            for in_rank in in_attribute['rank']:
                # compute the primitive name taking into account
                # the right category, dim and rank
                in_rank = in_rank.replace("SC", "")
                primitive_name = general_primitive['name'] + \
                    in_category + in_dim + in_rank
                in_type_name = []
                # compute the input type list
                for i, input in enumerate(primitive['input']):
                    # float type must be handled separately
                    if input == "float":
                        in_type_name.append(input)
                    elif len(in_rank) == 2:
                        # in this case the correct rank must be taken
                        in_type_name.append(input + in_category +
                                            in_dim + in_rank[i])
                    else:
                        in_type_name.append(input + in_category + in_dim + in_rank)
                in_type = list(map(eval, in_type_name))
                out_category = map_rule['category'](in_category)
                out_dim = str(map_rule['dimension'](int(in_dim)))
                out_rank = map_rule['rank'](in_rank)
                out_type_name = primitive['output'] + out_category + out_dim + out_rank
                out_type = eval(out_type_name)
                primitive_dictionary[primitive_name] = PrimitiveParams(
                    general_primitive['fun'], in_type, out_type)
    return primitive_dictionary


# define scalar primitives
scalar_primitives = {
    # scalar operations
    'add': PrimitiveParams(np.add, [float, float], float),
    'sub': PrimitiveParams(np.subtract, [float, float], float),
    'mul': PrimitiveParams(np.multiply, [float, float], float),
    'div': PrimitiveParams(np.divide, [float, float], float),
    'sin': PrimitiveParams(np.sin, [float], float),
    'arcsin': PrimitiveParams(np.arcsin, [float], float),
    'cos': PrimitiveParams(np.cos, [float], float),
    'arccos': PrimitiveParams(np.arccos, [float], float),
    'exp': PrimitiveParams(np.exp, [float], float),
    'log': PrimitiveParams(np.log, [float], float),
    'sqrt': PrimitiveParams(np.sqrt, [float], float),
    'square': PrimitiveParams(np.square, [float], float),
    # scalar operations (JAX backend)
    'AddF': PrimitiveParams(jnp.add, [float, float], float),
    'SubF': PrimitiveParams(jnp.subtract, [float, float], float),
    'MulF': PrimitiveParams(jnp.multiply, [float, float], float),
    'Div': PrimitiveParams(protectedDiv, [float, float], float),
    'SinF': PrimitiveParams(jnp.sin, [float], float),
    'ArcsinF': PrimitiveParams(jnp.arcsin, [float], float),
    'CosF': PrimitiveParams(jnp.cos, [float], float),
    'ArccosF': PrimitiveParams(jnp.arccos, [float], float),
    'ExpF': PrimitiveParams(jnp.exp, [float], float),
    'LogF': PrimitiveParams(protectedLog, [float], float),
    'SqrtF': PrimitiveParams(protectedSqrt, [float], float),
    'SquareF': PrimitiveParams(jnp.square, [float], float),
    'InvF': PrimitiveParams(inv_float, [float], float)}

# define cochain primitives
add_coch = {'fun_info': {'name': 'AddC', 'fun': C.add},
            'input': ["C.Cochain", "C.Cochain"],
            'output': "C.Cochain",
            'att_input': {'category': ('P', 'D'), 'dimension': ('0', '1', '2'),
                          "rank": ("SC", "V", "T")},
            'map_rule': {'category': lambda x: x, 'dimension': lambda x: x,
                         "rank": lambda x: x}}
sub_coch = {'fun_info': {'name': 'SubC', 'fun': C.sub},
            'input': ["C.Cochain", "C.Cochain"],
            'output': "C.Cochain",
            'att_input': {'category': ('P', 'D'), 'dimension': ('0', '1', '2'),
                          "rank": ("SC", "V", "T")},
            'map_rule': {'category': lambda x: x, 'dimension': lambda x: x,
                         "rank": lambda x: x}}
coboundary = {'fun_info': {'name': 'cob', 'fun': C.coboundary},
              'input': ["C.Cochain"],
              'output': "C.Cochain",
              'att_input': {'category': ('P', 'D'), 'dimension': ('0', '1'),
                            "rank": ("SC", "V", "T")},
              'map_rule': {'category': lambda x: x,
                           'dimension': partial(operator.add, 1),
                           "rank": lambda x: x}}
codifferential = {'fun_info': {'name': 'del', 'fun': C.codifferential},
                  'input': ["C.Cochain"],
                  'output': "C.Cochain",
                  'att_input': {'category': ('P', 'D'), 'dimension': ('1', '2'),
                                "rank": ("SC", "V", "T")},
                  'map_rule': {'category': lambda x: x, 'dimension':
                               partial(operator.add, -1), "rank": lambda x: x}}
tr_coch = {'fun_info': {'name': 'tr', 'fun': C.trace},
           'input': ["C.Cochain"],
           'output': "C.Cochain",
           'att_input': {'category': ('P', 'D'), 'dimension': ('0', '1', '2'),
                         "rank": ("T",)},
           'map_rule': {'category': lambda x: x, 'dimension': lambda x: x,
                        "rank": lambda x: ""}}
mul_FT = {'fun_info': {'name': 'MF', 'fun': C.scalar_mul},
          'input': ["C.Cochain", "float"],
          'output': "C.Cochain",
          'att_input': {'category': ('P', 'D'), 'dimension': ('0', '1', '2'),
                        "rank": ("SC", "V", "T")},
          'map_rule': {'category': lambda x: x, 'dimension': lambda x: x,
                       "rank": lambda x: x}}
inv_mul_FT = {'fun_info': {'name': 'InvM', 'fun': inv_scalar_mul},
              'input': ["C.Cochain", "float"],
              'output': "C.Cochain",
              'att_input': {'category': ('P', 'D'), 'dimension': ('0', '1', '2'),
                            "rank": ("SC", "V", "T")},
              'map_rule': {'category': lambda x: x, 'dimension': lambda x: x,
                           "rank": lambda x: x}}
mul_coch = {'fun_info': {'name': 'CMul', 'fun': C.cochain_mul},
            'input': ["C.Cochain", "C.Cochain"],
            'output': "C.Cochain",
            'att_input': {'category': ('P', 'D'), 'dimension': ('0', '1', '2'),
                          "rank": ("SC",)},
            'map_rule': {'category': lambda x: x, 'dimension': lambda x: x,
                         "rank": lambda x: x}}
# mul_VT = {'fun_info': {'name': 'Mv', 'fun': C.tensor_coch_mul},
#           'input': ["C.Cochain", "C.Cochain"],
#           'output': "C.Cochain",
#           'att_input': {'category': ('P', 'D'), 'dimension': ('0', '1', '2'),
#                         "rank": ("ST", "VT")},
#           'map_rule': {'category': lambda x: x, 'dimension': lambda x: x,
#                        "rank": lambda x: "T"}}
tran_coch = {'fun_info': {'name': 'tran', 'fun': C.transpose},
             'input': ["C.Cochain"],
             'output': "C.Cochain",
             'att_input': {'category': ('P', 'D'), 'dimension': ('0', '1', '2'),
                           "rank": ("T",)},
             'map_rule': {'category': lambda x: x, 'dimension': lambda x: x,
                          "rank": lambda x: x}}
sym_coch = {'fun_info': {'name': 'sym', 'fun': C.sym},
            'input': ["C.Cochain"],
            'output': "C.Cochain",
            'att_input': {'category': ('P', 'D'), 'dimension': ('0', '1', '2'),
                          "rank": ("T",)},
            'map_rule': {'category': lambda x: x, 'dimension': lambda x: x,
                         "rank": lambda x: x}}
star_1 = {'fun_info': {'name': 'St1', 'fun': C.star},
          'input': ["C.Cochain"],
          'output': "C.Cochain",
          'att_input': {'category': ('P', 'D'), 'dimension': ('0', '1'),
                        "rank": ("SC", "V", "T")},
          'map_rule': {'category': partial(switch_category, ('P', 'D')),
                       'dimension': partial(lambda x, y: y - x, y=1),
                       "rank": lambda x: x}}
star_2 = {'fun_info': {'name': 'St2', 'fun': C.star},
          'input': ["C.Cochain"],
          'output': "C.Cochain",
          'att_input': {'category': ('P', 'D'), 'dimension': ('0', '1', '2'),
                        "rank": ("SC", "V", "T")},
          'map_rule': {'category': partial(switch_category, ('P', 'D')),
                       'dimension': partial(lambda x, y: y-x, y=2),
                       "rank": lambda x: x}}
inner_product = {'fun_info': {'name': 'Inn', 'fun': C.inner},
                 'input': ["C.Cochain", "C.Cochain"],
                 'output': "float",
                 'att_input': {'category': ('P', 'D'), 'dimension': ('0', '1', '2'),
                               "rank": ("SC", "V", "T")},
                 'map_rule': {'category': lambda x: "", 'dimension': lambda x: "",
                              "rank": lambda x: ""}}
sin_coch = {'fun_info': {'name': 'Sin', 'fun': C.sin},
            'input': ["C.Cochain"],
            'output': "C.Cochain",
            'att_input': {'category': ('P', 'D'), 'dimension': ('0', '1', '2'),
                          "rank": ("SC", "V", "T")},
            'map_rule': {'category': lambda x: x, 'dimension': lambda x: x,
                         "rank": lambda x: x}}
arcsin_coch = {'fun_info': {'name': 'ArcSin', 'fun': C.arcsin},
               'input': ["C.Cochain"],
               'output': "C.Cochain",
               'att_input': {'category': ('P', 'D'), 'dimension': ('0', '1', '2'),
                             "rank": ("SC", "V", "T")},
               'map_rule': {'category': lambda x: x, 'dimension': lambda x: x,
                            "rank": lambda x: x}}
cos_coch = {'fun_info': {'name': 'Cos', 'fun': C.cos},
            'input': ["C.Cochain"],
            'output': "C.Cochain",
            'att_input': {'category': ('P', 'D'), 'dimension': ('0', '1', '2'),
                          "rank": ("SC", "V", "T")},
            'map_rule': {'category': lambda x: x, 'dimension': lambda x: x,
                         "rank": lambda x: x}}
arccos_coch = {'fun_info': {'name': 'ArcCos', 'fun': C.arccos},
               'input': ["C.Cochain"],
               'output': "C.Cochain",
               'att_input': {'category': ('P', 'D'), 'dimension': ('0', '1', '2'),
                             "rank": ("SC", "V", "T")},
               'map_rule': {'category': lambda x: x, 'dimension': lambda x: x,
                            "rank": lambda x: x}}
exp_coch = {'fun_info': {'name': 'Exp', 'fun': C.exp},
            'input': ["C.Cochain"],
            'output': "C.Cochain",
            'att_input': {'category': ('P', 'D'), 'dimension': ('0', '1', '2'),
                          "rank": ("SC", "V", "T")},
            'map_rule': {'category': lambda x: x, 'dimension': lambda x: x,
                         "rank": lambda x: x}}
log_coch = {'fun_info': {'name': 'Log', 'fun': C.log},
            'input': ["C.Cochain"],
            'output': "C.Cochain",
            'att_input': {'category': ('P', 'D'), 'dimension': ('0', '1', '2'),
                          "rank": ("SC", "V", "T")},
            'map_rule': {'category': lambda x: x, 'dimension': lambda x: x,
                         "rank": lambda x: x}}
sqrt_coch = {'fun_info': {'name': 'Sqrt', 'fun': C.sqrt},
             'input': ["C.Cochain"],
             'output': "C.Cochain",
             'att_input': {'category': ('P', 'D'), 'dimension': ('0', '1', '2'),
                           "rank": ("SC", "V", "T")},
             'map_rule': {'category': lambda x: x, 'dimension': lambda x: x,
                          "rank": lambda x: x}}
square_coch = {'fun_info': {'name': 'Square', 'fun': C.square},
               'input': ["C.Cochain"],
               'output': "C.Cochain",
               'att_input': {'category': ('P', 'D'), 'dimension': ('0', '1', '2'),
                             "rank": ("SC", "V", "T")},
               'map_rule': {'category': lambda x: x, 'dimension': lambda x: x,
                            "rank": lambda x: x}}
# def_gradient = {'fun_info': {'name': 'def_grad', 'fun': C.deformation_gradient},
#                 'input': ["C.Cochain"],
#                 'output': "C.Cochain",
#                 'att_input': {'category': ('P'), 'dimension': ('0'),
#                               "rank": ("V",)},
#                 'map_rule': {'category': partial(switch_category, ('P', 'D')),
#                              'dimension': lambda x: x, "rank": lambda x: "T"}}
coch_prim_list = [add_coch, sub_coch, coboundary, codifferential, tr_coch, mul_FT,
                  inv_mul_FT, mul_coch, tran_coch, sym_coch, star_1, star_2,
                  inner_product, sin_coch, arcsin_coch, cos_coch, arccos_coch, exp_coch,
                  log_coch, sqrt_coch, square_coch]
coch_primitives = list(map(generate_primitive, coch_prim_list))

# merge dictionary
primitives = scalar_primitives | {k: v for d in coch_primitives for k, v in d.items()}


def addPrimitivesToPset(pset: gp.PrimitiveSetTyped, pset_primitives: List = primitives,
                        new_primitives: Dict = {}) -> None:
    """Add a given list of primitives to a given PrimitiveSet.

    Args:
        pset: a primitive set.
        pset_primitives: list of primitives to be added. Each primitive is encoded
            as a dictionary composed of three keys: 'name', containing the name of
            the general primitive (e.g. cob for the coboundary); dimension', containing
            a list of the possible dimensions of the primitive input (or None if a
            scalar primitive is considered); 'rank', containing a list of the possible
            ranks of the primitive input (or None if a scalar primitive is considered).
    """
    full_primitives = primitives | {k: v for d in new_primitives for k, v in d.items()}
    for primitive in pset_primitives:
        # pre-process scalar primitives
        if primitive['dimension'] is None:
            primitive['dimension'] = []
        if primitive['rank'] is None:
            primitive['rank'] = []
        # save dimensions and ranks not admitted for the problem
        non_feasible_dimensions = list(set(('0', '1', '2')) -
                                       set(primitive['dimension']))
        non_feasible_ranks = list(
            set(("SC", "V", "T")) - set(primitive["rank"]))
        # iterate over all the primitives, pre-computed and stored in the dictionary
        # primitives
        for typed_primitive in full_primitives.keys():
            if primitive['name'] in typed_primitive:
                # remove the case in which the name of the primitive is a subname
                # of type_primitive (e.g. if primitive['name'] = sin and typed_primitive
                # = arcsin, we don't want to add the primitive)
                exact_name_check = len(
                    typed_primitive.replace(primitive['name'], "")) <= 2
                # check if the dimension/rank of a typed primitive
                # is admissible, i.e. if it does not coincide with a non-admissible
                # dimension/rank
                # FIXME: change this!
                check_wrong_dim_primal = sum([typed_primitive.count("P" + obj)
                                              for obj in non_feasible_dimensions])
                check_wrong_dim_dual = sum([typed_primitive.count("D" + obj)
                                            for obj in non_feasible_dimensions])
                check_rank = sum([typed_primitive.count("P" + obj)
                                  for obj in non_feasible_ranks])
                check_wrong_dim_rank = check_wrong_dim_primal + check_wrong_dim_dual +\
                    check_rank
                if check_wrong_dim_rank == 0 and exact_name_check:
                    op = full_primitives[typed_primitive].op
                    in_types = full_primitives[typed_primitive].in_types
                    out_type = full_primitives[typed_primitive].out_type
                    pset.addPrimitive(op, in_types, out_type, name=typed_primitive)
