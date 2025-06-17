from typing import Any, Callable
from functools import lru_cache

import jax
import jax.numpy as jnp
from pydantic import BaseModel, Field

from tesseract_core.runtime import Differentiable, Array, ShapeDType, Float32
from tesseract_core.runtime.tree_transforms import filter_func, flatten_with_paths

# Import JAX-FEM specific modules
from jax_fem.problem import Problem
from jax_fem.solver import ad_wrapper
from jax_fem.generate_mesh import get_meshio_cell_type, Mesh, rectangle_mesh

#
# Schemata
#


class InputSchema(BaseModel):
    rho: Differentiable[
        Array[
            (
                None,
                None,
            ),
            Float32,
        ]
    ] = Field(description="2D density field for topology optimization")

    Lx: float = Field(
        default=60.0, description="Length of the simulation box in the x direction."
    )
    Ly: float = Field(
        default=30.0,
        description="Length of the plane in the y direction. This is a scalar value that defines the size of the plane along the y-axis.",
    )
    Nx: int = Field(
        default=60,
        description="Number of elements in the x direction. This is an integer value that defines the resolution of the mesh along the x-axis.",
    )
    Ny: int = Field(
        default=30,
        description="Number of elements in the y direction. This is an integer value that defines the resolution of the mesh along the y-axis.",
    )


class OutputSchema(BaseModel):
    compliance: Differentiable[
        Array[
            (),
            Float32,
        ]
    ] = Field(description="Compliance of the structure, a measure of stiffness")


#
# Helper functions
#


# Define constitutive relationship
# Adapted from JAX-FEM
# https://github.com/deepmodeling/jax-fem/blob/1bdbf060bb32951d04ed9848c238c9a470fee1b4/demos/topology_optimization/example.py
class Elasticity(Problem):
    def custom_init(self):
        self.fe = self.fes[0]
        self.fe.flex_inds = jnp.arange(len(self.fe.cells))

    def get_tensor_map(self):
        def stress(u_grad, theta):
            Emax = 70.0e3
            Emin = 1e-3 * Emax
            nu = 0.3
            penal = 3.0
            E = Emin + (Emax - Emin) * theta[0] ** penal
            epsilon = 0.5 * (u_grad + u_grad.T)
            eps11 = epsilon[0, 0]
            eps22 = epsilon[1, 1]
            eps12 = epsilon[0, 1]
            sig11 = E / (1 + nu) / (1 - nu) * (eps11 + nu * eps22)
            sig22 = E / (1 + nu) / (1 - nu) * (nu * eps11 + eps22)
            sig12 = E / (1 + nu) * eps12
            sigma = jnp.array([[sig11, sig12], [sig12, sig22]])
            return sigma

        return stress

    def get_surface_maps(self):
        def surface_map(u, x):
            return jnp.array([0.0, 100.0])

        return [surface_map]

    def set_params(self, params):
        # Override base class method.
        full_params = jnp.ones((self.fe.num_cells, params.shape[1]))
        full_params = full_params.at[self.fe.flex_inds].set(params)
        thetas = jnp.repeat(full_params[:, None, :], self.fe.num_quads, axis=1)
        self.full_params = full_params
        self.internal_vars = [thetas]

    def compute_compliance(self, sol):
        # Surface integral
        boundary_inds = self.boundary_inds_list[0]
        _, nanson_scale = self.fe.get_face_shape_grads(boundary_inds)
        u_face = (
            sol[self.fe.cells][boundary_inds[:, 0]][:, None, :, :]
            * self.fe.face_shape_vals[boundary_inds[:, 1]][:, :, :, None]
        )
        u_face = jnp.sum(u_face, axis=2)
        subset_quad_points = self.physical_surface_quad_points[0]
        neumann_fn = self.get_surface_maps()[0]
        traction = -jax.vmap(jax.vmap(neumann_fn))(u_face, subset_quad_points)
        val = jnp.sum(traction * u_face * nanson_scale[:, :, None])
        return val


# Memoize the setup function to avoid expensive recomputation
@lru_cache(maxsize=1)
def setup(
    Nx: int = 60,
    Ny: int = 30,
    Lx: float = 60.0,
    Ly: float = 30.0,
) -> tuple[Elasticity, Callable]:
    # Specify mesh-related information. We use first-order quadrilateral element.
    ele_type = "QUAD4"
    cell_type = get_meshio_cell_type(ele_type)
    meshio_mesh = rectangle_mesh(Nx=Nx, Ny=Ny, domain_x=Lx, domain_y=Ly)
    mesh = Mesh(meshio_mesh.points, meshio_mesh.cells_dict[cell_type])

    # Define boundary conditions and values.
    def fixed_location(point):
        return jnp.isclose(point[0], 0.0, atol=1e-5)

    def load_location(point):
        return jnp.logical_and(
            jnp.isclose(point[0], Lx, atol=1e-5),
            jnp.isclose(point[1], 0.0, atol=0.1 * Ly + 1e-5),
        )

    def dirichlet_val(point):
        return 0.0

    dirichlet_bc_info = [[fixed_location] * 2, [0, 1], [dirichlet_val] * 2]

    location_fns = [load_location]

    # Define forward problem
    problem = Elasticity(
        mesh,
        vec=2,
        dim=2,
        ele_type=ele_type,
        dirichlet_bc_info=dirichlet_bc_info,
        location_fns=location_fns,
    )

    # Apply the automatic differentiation wrapper
    # This is a critical step that makes the problem solver differentiable
    fwd_pred = ad_wrapper(
        problem,
        solver_options={"umfpack_solver": {}},
        adjoint_solver_options={"umfpack_solver": {}},
    )
    return problem, fwd_pred


def apply_fn(inputs: dict) -> dict:
    """
    Compute the compliance of the structure given a density field.
    """
    problem, fwd_pred = setup(
        Nx=inputs["Nx"],
        Ny=inputs["Ny"],
        Lx=inputs["Lx"],
        Ly=inputs["Ly"],
    )
    rho = inputs["rho"]
    sol_list = fwd_pred(rho)
    compliance = problem.compute_compliance(sol_list[0])
    return {"compliance": compliance.astype(jnp.float32)}


#
# Tesseract endpoints
#


def apply(inputs: InputSchema) -> OutputSchema:
    return apply_fn(inputs.model_dump())


def vector_jacobian_product(
    inputs: InputSchema,
    vjp_inputs: set[str],
    vjp_outputs: set[str],
    cotangent_vector: dict[str, Any],
):
    assert vjp_inputs == {"rho"}
    assert vjp_outputs == {"compliance"}

    inputs = inputs.model_dump()

    filtered_apply = filter_func(apply_fn, inputs, vjp_outputs)
    _, vjp_func = jax.vjp(
        filtered_apply, flatten_with_paths(inputs, include_paths=vjp_inputs)
    )
    out = vjp_func(cotangent_vector)[0]
    return out


def abstract_eval(abstract_inputs):
    """Calculate output shape of apply from the shape of its inputs."""

    return {"compliance": ShapeDType(shape=(), dtype="float32")}
