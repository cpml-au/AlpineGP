import jax.numpy as jnp
from deap import gp
from dctkit.dec import cochain as C
from dctkit.dec import vector as V
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


def identity(x):
    return x


def empty_string(x):
    return ""


def rank_downgrade(x):
    if x == "T":
        return "V"
    elif x == "V":
        return ""
    raise ValueError("Invalid input rank")


def vec_tensor_mul_rank(x):
    return "T"


def star_dim(x, max_dim):
    return max_dim - x


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


scalar_primitives = {
    # scalar operations
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
    'SquareF': PrimitiveParams(square_mod, [float], float),
    'InvF': PrimitiveParams(inv_float, [float], float)}
coch_primitives = []
primitives = scalar_primitives
add_coch = {'fun_info': {'name': 'AddC', 'fun': C.add},
            'input': ["C.Cochain", "C.Cochain"],
            'output': "C.Cochain",
            'att_input': {'category': ('P', 'D'), 'dimension': ('0', '1', '2'),
                          "rank": ("SC", "V", "T")},
            'map_rule': {'category': identity, 'dimension': identity, "rank": identity}}
coch_primitives.append(generate_primitive(add_coch))
sub_coch = {'fun_info': {'name': 'SubC', 'fun': C.sub},
            'input': ["C.Cochain", "C.Cochain"],
            'output': "C.Cochain",
            'att_input': {'category': ('P', 'D'), 'dimension': ('0', '1', '2'),
                          "rank": ("SC", "V", "T")},
            'map_rule': {'category': identity, 'dimension': identity, "rank": identity}}
coch_primitives.append(generate_primitive(sub_coch))
coboundary = {'fun_info': {'name': 'cob', 'fun': C.coboundary},
              'input': ["C.Cochain"],
              'output': "C.Cochain",
              'att_input': {'category': ('P', 'D'), 'dimension': ('0', '1'),
                            "rank": ("SC",)},
              'map_rule': {'category': identity, 'dimension': partial(operator.add, 1),
                           "rank": identity}}
coch_primitives.append(generate_primitive(coboundary))
codifferential = {'fun_info': {'name': 'del', 'fun': C.codifferential},
                  'input': ["C.Cochain"],
                  'output': "C.Cochain",
                  'att_input': {'category': ('P', 'D'), 'dimension': ('1', '2'),
                                "rank": ("SC",)},
                  'map_rule': {'category': identity, 'dimension':
                               partial(operator.add, -1), "rank": identity}}
coch_primitives.append(generate_primitive(codifferential))
tr_coch = {'fun_info': {'name': 'tr', 'fun': C.trace},
           'input': ["C.Cochain"],
           'output': "C.Cochain",
           'att_input': {'category': ('P', 'D'), 'dimension': ('0', '1', '2'),
                         "rank": ("T",)},
           'map_rule': {'category': identity, 'dimension': identity,
                        "rank": rank_downgrade}}
coch_primitives.append(generate_primitive(tr_coch))
mul_FT = {'fun_info': {'name': 'MF', 'fun': C.scalar_mul},
          'input': ["C.Cochain", "float"],
          'output': "C.Cochain",
          'att_input': {'category': ('P', 'D'), 'dimension': ('0', '1', '2'),
                        "rank": ("SC", "T")},
          'map_rule': {'category': identity, 'dimension': identity, "rank": identity}}
coch_primitives.append(generate_primitive(mul_FT))
inv_mul_FT = {'fun_info': {'name': 'InvM', 'fun': inv_scalar_mul},
              'input': ["C.Cochain", "float"],
              'output': "C.Cochain",
              'att_input': {'category': ('P', 'D'), 'dimension': ('0', '1', '2'),
                            "rank": ("SC", "T")},
              'map_rule': {'category': identity, 'dimension': identity,
                           "rank": identity}}
coch_primitives.append(generate_primitive(inv_mul_FT))
mul_coch = {'fun_info': {'name': 'CMul', 'fun': C.cochain_mul},
            'input': ["C.Cochain", "C.Cochain"],
            'output': "C.Cochain",
            'att_input': {'category': ('P', 'D'), 'dimension': ('0', '1', '2'),
                          "rank": ("SC",)},
            'map_rule': {'category': identity, 'dimension': identity,
                         "rank": identity}}
coch_primitives.append(generate_primitive(mul_coch))
mul_VT = {'fun_info': {'name': 'Mv', 'fun': C.vector_tensor_mul},
          'input': ["C.Cochain", "C.Cochain"],
          'output': "C.Cochain",
          'att_input': {'category': ('P', 'D'), 'dimension': ('0', '1', '2'),
                        "rank": ("VT",)},
          'map_rule': {'category': identity, 'dimension': identity,
                       "rank": vec_tensor_mul_rank}}
coch_primitives.append(generate_primitive(mul_VT))
tran_coch = {'fun_info': {'name': 'tran', 'fun': C.transpose},
             'input': ["C.Cochain"],
             'output': "C.Cochain",
             'att_input': {'category': ('P', 'D'), 'dimension': ('0', '1', '2'),
                           "rank": ("T",)},
             'map_rule': {'category': identity, 'dimension': identity,
                          "rank": identity}}
coch_primitives.append(generate_primitive(tran_coch))
sym_coch = {'fun_info': {'name': 'sym', 'fun': C.sym},
            'input': ["C.Cochain"],
            'output': "C.Cochain",
            'att_input': {'category': ('P', 'D'), 'dimension': ('0', '1', '2'),
                          "rank": ("T",)},
            'map_rule': {'category': identity, 'dimension': identity, "rank": identity}}
coch_primitives.append(generate_primitive(sym_coch))
star_1 = {'fun_info': {'name': 'St1', 'fun': C.star},
          'input': ["C.Cochain"],
          'output': "C.Cochain",
          'att_input': {'category': ('P', 'D'), 'dimension': ('0', '1'),
                        "rank": ("SC", "T")},
          'map_rule': {'category': partial(switch_category, ('P', 'D')),
                       'dimension': partial(star_dim, max_dim=1), "rank": identity}}
coch_primitives.append(generate_primitive(star_1))
star_2 = {'fun_info': {'name': 'St2', 'fun': C.star},
          'input': ["C.Cochain"],
          'output': "C.Cochain",
          'att_input': {'category': ('P', 'D'), 'dimension': ('0', '1', '2'),
                        "rank": ("SC", "T")},
          'map_rule': {'category': partial(switch_category, ('P', 'D')),
                       'dimension': partial(star_dim, max_dim=2), "rank": identity}}
coch_primitives.append(generate_primitive(star_2))
inner_product = {'fun_info': {'name': 'Inn', 'fun': C.inner_product},
                 'input': ["C.Cochain", "C.Cochain"],
                 'output': "float",
                 'att_input': {'category': ('P', 'D'), 'dimension': ('0', '1', '2'),
                               "rank": ("SC", "T")},
                 'map_rule': {'category': empty_string, 'dimension': empty_string,
                              "rank": empty_string}}
coch_primitives.append(generate_primitive(inner_product))
sin_coch = {'fun_info': {'name': 'Sin', 'fun': C.sin},
            'input': ["C.Cochain"],
            'output': "C.Cochain",
            'att_input': {'category': ('P', 'D'), 'dimension': ('0', '1', '2'),
                          "rank": ("SC",)},
            'map_rule': {'category': identity, 'dimension': identity,
                         "rank": identity}}
coch_primitives.append(generate_primitive(sin_coch))
arcsin_coch = {'fun_info': {'name': 'ArcSin', 'fun': C.arcsin},
               'input': ["C.Cochain"],
               'output': "C.Cochain",
               'att_input': {'category': ('P', 'D'), 'dimension': ('0', '1', '2'),
                             "rank": ("SC",)},
               'map_rule': {'category': identity, 'dimension': identity,
                            "rank": identity}}
coch_primitives.append(generate_primitive(arcsin_coch))
cos_coch = {'fun_info': {'name': 'Cos', 'fun': C.cos},
            'input': ["C.Cochain"],
            'output': "C.Cochain",
            'att_input': {'category': ('P', 'D'), 'dimension': ('0', '1', '2'),
                          "rank": ("SC",)},
            'map_rule': {'category': identity, 'dimension': identity,
                         "rank": identity}}
coch_primitives.append(generate_primitive(cos_coch))
arccos_coch = {'fun_info': {'name': 'ArcCos', 'fun': C.arccos},
               'input': ["C.Cochain"],
               'output': "C.Cochain",
               'att_input': {'category': ('P', 'D'), 'dimension': ('0', '1', '2'),
                             "rank": ("SC",)},
               'map_rule': {'category': identity, 'dimension': identity,
                            "rank": identity}}
coch_primitives.append(generate_primitive(arccos_coch))
exp_coch = {'fun_info': {'name': 'Exp', 'fun': C.exp},
            'input': ["C.Cochain"],
            'output': "C.Cochain",
            'att_input': {'category': ('P', 'D'), 'dimension': ('0', '1', '2'),
                          "rank": ("SC",)},
            'map_rule': {'category': identity, 'dimension': identity, "rank": identity}}
coch_primitives.append(generate_primitive(exp_coch))
log_coch = {'fun_info': {'name': 'Log', 'fun': C.log},
            'input': ["C.Cochain"],
            'output': "C.Cochain",
            'att_input': {'category': ('P', 'D'), 'dimension': ('0', '1', '2'),
                          "rank": ("SC",)},
            'map_rule': {'category': identity, 'dimension': identity, "rank": identity}}
coch_primitives.append(generate_primitive(log_coch))
sqrt_coch = {'fun_info': {'name': 'Sqrt', 'fun': C.sqrt},
             'input': ["C.Cochain"],
             'output': "C.Cochain",
             'att_input': {'category': ('P', 'D'), 'dimension': ('0', '1', '2'),
                           "rank": ("SC",)},
             'map_rule': {'category': identity, 'dimension': identity,
                          "rank": identity}}
coch_primitives.append(generate_primitive(sqrt_coch))
square_coch = {'fun_info': {'name': 'Square', 'fun': C.square},
               'input': ["C.Cochain"],
               'output': "C.Cochain",
               'att_input': {'category': ('P', 'D'), 'dimension': ('0', '1', '2'),
                             "rank": ("SC",)},
               'map_rule': {'category': identity, 'dimension': identity,
                            "rank": identity}}
coch_primitives.append(generate_primitive(square_coch))
flat_up = {'fun_info': {'name': 'flat_up',
                        'fun': partial(V.flat_PDD, scheme="upwind")},
           'input': ["C.Cochain"],
           'output': "C.Cochain",
           'att_input': {'category': ('D',), 'dimension': ('0',),
                         "rank": ("SC",)},
           'map_rule': {'category': identity, 'dimension': partial(operator.add, 1),
                        "rank": identity}}
coch_primitives.append(generate_primitive(flat_up))
flat_par = {'fun_info': {'name': 'flat_par',
                         'fun': partial(V.flat_PDD, scheme="parabolic")},
            'input': ["C.Cochain"],
            'output': "C.Cochain",
            'att_input': {'category': ('D',), 'dimension': ('0',),
                          "rank": ("SC",)},
            'map_rule': {'category': identity, 'dimension': partial(operator.add, 1),
                         "rank": identity}}
coch_primitives.append(generate_primitive(flat_par))
# FIXME: extend auto-generator to handle scalar fields primitives
coch_primitives.append({'inter_up': PrimitiveParams(
    V.upwind_interpolation, [C.CochainD0], V.ScalarField)})
coch_primitives.append({'int_up': PrimitiveParams(
    V.upwind_integration, [V.ScalarField], C.CochainD1)})


for primitive in coch_primitives:
    # merge dictionary
    primitives = primitives | primitive
primitives = scalar_primitives | primitives


def addPrimitivesToPset(pset: gp.PrimitiveSetTyped, pset_primitives: List) -> None:
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
            set(("SC", "V", "T", "vtm")) - set(primitive["rank"]))
        non_feasible_objects = non_feasible_dimensions + non_feasible_ranks
        # iterate over all the primitives, pre-computed and stored in the dictionary
        # primitives
        for typed_primitive in primitives.keys():
            if primitive['name'] in typed_primitive:
                # check if the dimension/rank of a typed primitive
                # is admissible, i.e. if it does not coincide with a non-admissible
                # dimension/rank
                if sum([typed_primitive.count(obj)
                        for obj in non_feasible_objects]) == 0 or \
                        typed_primitive.count("VT") == 1:
                    op = primitives[typed_primitive].op
                    in_types = primitives[typed_primitive].in_types
                    out_type = primitives[typed_primitive].out_type
                    pset.addPrimitive(op, in_types, out_type, name=typed_primitive)
