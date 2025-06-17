from typing import Any
import numpy as np
from pydantic import BaseModel, Field
import pyvista as pv

from tesseract_core.runtime import Differentiable, Array, ShapeDType, Float32

#
# Schemata
#


class InputSchema(BaseModel):
    bar_params: Differentiable[
        Array[
            (None, None, 3),
            Float32,
        ]
    ] = Field(
        description="""Vertex positions of the bar geometry. 
        The shape is (num_bars, num_vertices, 3), where num_bars is the number of bars and num_vertices is the number of vertices per bar. The last dimension represents the x, y, z coordinates of each vertex."""
    )

    bar_radius: float = Field(
        default=1.5,
        description="Radius of the bars in the geometry. This is a scalar value that defines the thickness of the bars.",
    )

    Lx: float = Field(
        default=60.0,
        description="Length of the plane in the x direction. This is a scalar value that defines the size of the plane along the x-axis.",
    )
    Ly: float = Field(
        default=30.0,
        description="Length of the plane in the y direction. This is a scalar value that defines the size of the plane along the y-axis.",
    )
    Nx: int = Field(
        default=60,
        description="Number of points in the x direction. This is an integer value that defines the resolution of the plane along the x-axis.",
    )
    Ny: int = Field(
        default=30,
        description="Number of points in the y direction. This is an integer value that defines the resolution of the plane along the y-axis.",
    )
    epsilon: float = Field(
        default=1e-5,
        description="Epsilon value for finite difference approximation of the Jacobian. This is a small scalar value used to compute the numerical gradient.",
    )


class OutputSchema(BaseModel):
    sdf: Differentiable[
        Array[
            (
                None,
                None,
            ),
            Float32,
        ]
    ] = Field(description="SDF field of the geometry")


#
# Helper functions
#


def build_geometry(
    params: np.ndarray,
    radius: float,
) -> list[pv.PolyData]:
    """
    Build a pyvista geometry from the parameters.

    The parameters are expected to be of shape (n_chains, n_edges_per_chain + 1, 3),
    """
    n_chains = params.shape[0]
    geometry = []

    for chain in range(n_chains):
        tube = pv.Spline(points=params[chain]).tube(radius=radius, capping=False)
        geometry.append(tube)

    return geometry


def compute_sdf(
    params: np.ndarray,
    radius: float,
    Lx: float,
    Ly: float,
    Nx: int,
    Ny: int,
) -> pv.PolyData:
    """Create a pyvista plane that has the SDF values stored as a vertex attribute.

    The SDF field is computed based on the geometry defined by the parameters.
    """
    grid_coords = pv.Plane(
        center=(0, 0, 0),
        direction=(0, 0, 1),
        i_size=Lx,
        j_size=Ly,
        i_resolution=Nx - 1,
        j_resolution=Ny - 1,
    )
    grid_coords = grid_coords.triangulate()

    geometries = build_geometry(
        params,
        radius=radius,
    )

    sdf_field = None

    for geometry in geometries:
        # Compute the implicit distance from the geometry to the grid coordinates.
        # The implicit distance is a signed distance field, where positive values are outside the geometry and negative values are inside.
        this_sdf = grid_coords.compute_implicit_distance(geometry.triangulate())
        if sdf_field is None:
            sdf_field = this_sdf
        else:
            sdf_field["implicit_distance"] = np.minimum(
                sdf_field["implicit_distance"], this_sdf["implicit_distance"]
            )

    return sdf_field


def apply_fn(
    params: np.ndarray,
    radius: float,
    Lx: float,
    Ly: float,
    Nx: int,
    Ny: int,
) -> np.ndarray:
    """
    Get the sdf values of a the geometry defined by the parameters as a 2D array.
    """
    sdf_geom = compute_sdf(
        params,
        radius=radius,
        Lx=Lx,
        Ly=Ly,
        Nx=Nx,
        Ny=Ny,
    )["implicit_distance"]

    # The implicit distance is a 1D where the indexing is tranposed.
    # We need to reshape it to a 2D array with the shape (Ny, Nx) and then transpose it to get the correct orientation.
    return sdf_geom.reshape((Ny, Nx)).T


def jac_sdf_wrt_params(
    params: np.ndarray,
    radius: float,
    Lx: float,
    Ly: float,
    Nx: int,
    Ny: int,
    epsilon: float,
) -> np.ndarray:
    """
    Compute the Jacobian of the SDF values with respect to the parameters.
    The Jacobian is computed by finite differences.
    The shape of the Jacobian is (n_chains, n_edges_per_chain + 1, 3, Nx, Ny).
    """

    n_chains = params.shape[0]
    n_edges_per_chain = params.shape[1] - 1

    jac = np.zeros(
        (
            n_chains,
            n_edges_per_chain + 1,
            3,  # number of dimensions (x, y, z)
            Nx,
            Ny,
        )
    )

    sdf_base = apply_fn(
        params,
        radius=radius,
        Lx=Lx,
        Ly=Ly,
        Nx=Nx,
        Ny=Ny,
    )

    for chain in range(n_chains):
        for vertex in range(0, n_edges_per_chain + 1):
            # we only care about the y coordinate
            i = 1
            params_eps = params.copy()
            params_eps[chain, vertex, i] += epsilon

            sdf_epsilon = apply_fn(
                params_eps,
                radius=radius,
                Lx=Lx,
                Ly=Ly,
                Nx=Nx,
                Ny=Ny,
            )
            jac[chain, vertex, i] = (sdf_epsilon - sdf_base) / epsilon

    return jac


#
# Tesseract endpoints
#


def apply(inputs: InputSchema) -> OutputSchema:
    return OutputSchema(
        sdf=apply_fn(
            inputs.bar_params,
            radius=inputs.bar_radius,
            Lx=inputs.Lx,
            Ly=inputs.Ly,
            Nx=inputs.Nx,
            Ny=inputs.Ny,
        )
    )


def vector_jacobian_product(
    inputs: InputSchema,
    vjp_inputs: set[str],
    vjp_outputs: set[str],
    cotangent_vector: dict[str, Any],
):
    assert vjp_inputs == {"bar_params"}
    assert vjp_outputs == {"sdf"}

    jac = jac_sdf_wrt_params(
        inputs.bar_params,
        radius=inputs.bar_radius,
        Lx=inputs.Lx,
        Ly=inputs.Ly,
        Nx=inputs.Nx,
        Ny=inputs.Ny,
        epsilon=inputs.epsilon,
    )
    # Reduce the cotangent vector to the shape of the Jacobian, to compute VJP by hand
    vjp = np.einsum("ijklm,lm->ijk", jac, cotangent_vector["sdf"]).astype(np.float32)
    return {"bar_params": vjp}


def abstract_eval(abstract_inputs):
    """Calculate output shape of apply from the shape of its inputs."""

    return {
        "sdf": ShapeDType(
            shape=(abstract_inputs.Nx, abstract_inputs.Ny), dtype="float32"
        )
    }
