from dctkit.mesh.simplex import SimplicialComplex
from dctkit.mesh.util import generate_1_D_mesh


def get_1D_complex(num_nodes: int, length: float) -> SimplicialComplex:
    S_1, x = generate_1_D_mesh(num_nodes=num_nodes, L=length)
    S = SimplicialComplex(S_1, x, is_well_centered=True)
    S.get_circumcenters()
    S.get_primal_volumes()
    S.get_dual_volumes()
    S.get_hodge_star()
    return S
