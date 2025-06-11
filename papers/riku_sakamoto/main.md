---
# Ensure that this title is the same as the one in `myst.yml`
title: "Phlower: A Deep Learning Framework Supporting PyTorch Tensors with Physical Dimensions"
abstract: |
  We created [Phlower](https://github.com/ricosjp/phlower), an open-source deep learning library that extends PyTorch tensors to support physical dimensions — such as time (T), mass (M), and length (L) — and enforces dimensional consistency throughout computations. In this paper, we will present Phlower’s core features and demonstrate how it ensures dimensional correctness in deep learning tasks.

---

## Introduction

The simulation of physical phenomena plays a critical role in various aspects of daily life, including vehicle design and weather forecasting. As a new paradigm, _physics-informed machine learning_ which integrates machine learning with physical simulations, has recently gained significant attention, particularly with the evolution of Physics-Informed Neural Networks (PINNs) [@RAISSI2019686] and Graph Neural Networks (GNNs). Compared to traditional simulation methods, it offers the potential to accelerate predictions of physical behavior and to uncover new physical laws that might otherwise remain hidden.

In numerical simulations of physical phenomena, it is essential to handle physical dimensions — such as time (T), mass (M), and length (L) — to ensure the correctness of computations. Extending this concept to deep learning, handling physical dimensions provides two main benefits. First, maintaining consistency of physical dimensions enhances the reliability of the model architecture. Second, physical dimensions provide a principled basis for scaling inputs to machine learning models.

To address these needs, we developed Phlower, an open-source deep learning framework that enables PyTorch [@NEURIPS2019_bdbca288] tensor objects to carry physical dimensions. An example illustrating its usage is shown in the [](#example_phlower_tensor).



```{code} python
:label: example_phlower_tensor
:caption: PhlowerTensor carrying physical dimensions

from phlower import phlower_tensor
import torch

# Example: Calculating kinetic energy (E = 0.5 * m * v^2)

# Assuming mass (m) has dimensions M^1
mass = phlower_tensor(torch.rand(10, 1), dimension={"M": 1})

# Assuming velocity (v) has dimensions L^1 T^-1
velocity = phlower_tensor(torch.rand(10, 3, 1), dimension={"L": 1, "T": -1})

kinetic_energy = 0.5 * mass * torch.sum(velocity ** 2, axis=-1)

print(kinetic_energy.dimension)
# Output: PhlowerDimensionTensor(T: -2.0, L: 2.0, M: 1.0, I: 0.0, Theta: 0.0, N: 0.0, J: 0.0)
```

In this paper, we will introduce Phlower’s key features and demonstrate how physical dimensions are handled in deep learning workflows. The remainder of this paper is organized as follows:

* Section 2 introduces the core concept of Phlower and its key features. Also, we introduce the basic usage of `PhlowerTensor`, which is a wrapper class of PyTorch [@NEURIPS2019_bdbca288] Tensor object and enables physical dimension tracking.

* Section 3 discusses the possible use cases of Phlower.

* Section 4 discusses additional features of Phlower, including YAML-based model definition and shape semantics.

* Section 5 compares Phlower with related work.


## Design and Core Concepts


### Motivation

PyTorch is a widely used deep learning framework that provides powerful tensor operations and automatic differentiation capabilities. However, it does not natively support physical dimensions, which are crucial for ensuring the correctness of computations in physics-related applications. This limitation can lead to errors and inconsistencies when performing operations on tensors representing physical quantities. Here, physical dimensions refer to the fundamental quantities that describe physical phenomena, such as time ( {math}`T` ), mass ( {math}`M` ), length ( {math}`L` ), electric current ( {math}`I` ), temperature ( {math}`\Theta` ), amount of substance ( {math}`N` ), and luminous intensity ( {math}`J` ). 

Therfore, Phlower introduces `PhlowerTensor` which is a wrapper class of PyTorch Tensor object and enables physical dimension tracking. This ensures that tensor operations adhere to dimensional consistency rules. Thanks to its lightweight implementation, converting between torch.Tensor and PhlowerTensor is straightforward. Moreover, PhlowerTensor maintains compatibility with PyTorch’s autograd system and core tensor operations such as `sum()`, making it easy to integrate into existing PyTorch-based workflows with minimal code modifications.



### Basic Usage

This section introduces the basic usage of `PhlowerTensor`.

#### Defining PhlowerTensor

[](#create_phlower_tensor) shows how to create a `PhlowerTensor` from a PyTorch tensor. The `phlower_tensor` function takes a PyTorch tensor and a dictionary specifying the physical dimensions. The dimensions are defined using the International System of Units (SI) base units, such as time (T), mass (M), and length (L).


```{code} python
:label: create_phlower_tensor
:caption: Create PhlowerTensor from PyTorch Tensor

from phlower import phlower_tensor
import torch

# Create a PhlowerTensor with physical dimensions at each point
# Example: A tensor representing velocity with dimensions L^1 T^-1
velocity = phlower_tensor(torch.rand(10, 3),  dimension={"L": 1, "T": -1})
print(velocity)

# Example: A tensor representing pressure with dimensions M^1 T^-2 L^-1
pressure = phlower_tensor(torch.rand(10, 1),  dimension={"M": 1, "T": -2, "L": -1})
print(pressure)

```


#### Dimensional Consistency

PhlowerTensor enforces dimensional consistency during tensor operations. [](#kinetic_energy_example) demonstrates how PhlowerTensor ensures dimensional consistency when computing kinetic energy, defined as {math}` E = 0.5 \cdot m \cdot \mathbf{v}^2 `, where {math}` m ` is mass and {math}` \mathbf{v} ` is velocity. This example assumes that there are 10 observing points, the velocity is a 3-dimensional vector (e.g., in 3D space. See [](#fig:fig_points)), and the mass is a scalar. The resulting kinetic energy will have dimensions {math}` M^1 L^2 T^{-2} `, corresponding to the physical dimension of energy.


:::{figure} images/figure1.png
:label: fig:fig_points
Example of point clouds. Each point has mass and velocity.
:::


```{code} python
:label: kinetic_energy_example
:caption: Calculating kinetic energy with PhlowerTensor

from phlower import phlower_tensor
import torch

# Example: Calculating kinetic energy (E = 0.5 * m * v^2)

# Assuming mass (m) has dimensions M^1
mass = phlower_tensor(torch.rand(10, 1), dimension={"M": 1})

# Assuming velocity (v) has dimensions L^1 T^-1
velocity = phlower_tensor(torch.rand(10, 3, 1), dimension={"L": 1, "T": -1})

kinetic_energy = 0.5 * mass * torch.sum(velocity ** 2, axis=-1)

print(kinetic_energy.dimension)
# Output: PhlowerDimensionTensor(T: -2.0, L: 2.0, M: 1.0, I: 0.0, Theta: 0.0, N: 0.0, J: 0.0)

```


On the other hand, Phlower disallows operations involving incompatible physical dimensions. [](#incompatible_dimensions_example) shows `PhlowerTensor` raise an error when attempting to add a pressure tensor to a velocity tensor.


```{code} python
:label: incompatible_dimensions_example
:caption: Attempting to add tensors with incompatible dimensions

# Example: A tensor representing velocity with dimensions L^1 T^-1
velocity = phlower_tensor(torch.rand(10, 3),  dimension={"L": 1, "T": -1})

# Example: A tensor representing pressure with dimensions M^1 T^-2 L^-1
pressure = phlower_tensor(torch.rand(10, 1),  dimension={"M": 1, "T": -2, "L": -1})

# Attempting to add velocity and pressure tensors with incompatible dimensions
try:
    result = velocity + pressure
except ValueError as e:
    print(f"Error: {e}")
    # Output: Error: Incompatible dimensions for addition: L^1 T^-1 vs M^1 T^-2 L^-1

```

#### Array Operations

This section demonstrates how `PhlowerTensor` supports array operations while maintaining dimensional consistency.
Stacking tensors with compatible dimensions is straightforward. For example, [](#-stacking_tensors_example) shows that multiple velocity tensors can be stacked along a new dimension.
The resulting tensor remains `PhlowerTensor` with appropriate physical dimension, even though the standard `torch.stack` is called.

```{code} python
:label: stacking_tensors_example
:caption: Stacking tensors with compatible dimensions

from phlower import phlower_tensor
import torch

# Example: Stacking velocity tensors with compatible dimensions
# Assuming we have multiple velocity tensors with dimensions L^1 T^-1
velocity1 = phlower_tensor(torch.rand(10, 3), dimension={"L": 1, "T": -1})
velocity2 = phlower_tensor(torch.rand(10, 3), dimension={"L": 1, "T": -1})

# Stacking along a new dimension (e.g., time)
stacked_velocity = torch.stack([velocity1, velocity2], dim=0), dimension={"L": 1, "T": -1}
print(stacked_velocity)
# Output: PhlowerTensor with dimensions L^1 T^-1 and shape (2, 10, 3)

```


## Use Cases

This section presents the use cases of PhlowerTensor in deep learning applications, particularly in the context of scientific machine learning and physics-informed models. PhlowerTensor's ability to track physical dimensions and enforce dimensional consistency makes it a valuable tool for ensuring the correctness of computations in these domains.


### Preventing Dimensional Inconsistencies in Physics-Informed Models

In physics-informed neural networks (PINNs), the loss function often includes terms derived from differential equations. For example, when modeling a system governed by Navier-Stokes equations, the loss function is composed of terms like the continuity equation and momentum equations. These equations involve derivatives with respect to spatial and temporal dimensions, which can lead to dimensional inconsistencies if not handled correctly.


### Enforcing Dimensional Validity in Feature Engineering

Phlower can reject invalid operations in feature engineering (e.g., subtracting pressure from velocity) due to its enforcement of dimensional consistency. This capability is particularly useful in domains where data includes a variety of physical quantities with different dimensions.


## Additional Features

### YAML-Based Definition

Phlower provides a YAML-based model definition system that allows users to define, configure, and reuse machine learning models efficiently. [](#phlower_yaml_example) shows an excerpt of an example YAML file. This feature simplifies experimentation by enabling users to modify model architectures and hyperparameters without changing the underlying code.

```{code} yaml
:label: phlower_yaml_example
:caption: Example of YAML-based model definition in Phlower

- nn_type: MLP
  name: ENCODER1
  input_keys:
    - feature1
  output_key: mlp1
  destinations:
    - Concat
  nn_parameters:
    nodes: [-1, 20, 200]
    activations: ["relu", "relu"]

```


### Shape Semantics

Each index of a `PhlowerTensor`'s shape carries semantic meaning. This concept is referred to as _shape semantics_ in Phlower. For example, if a tensor represents a velocity field in 3D space, the first dimension represents time steps when working with time-series data. [](#shape_semantics_example) shows an example of how `PhlowerTensor` can represent time-series pressure using shape semantics. This positional information is crucial for ensuring that operations on tensors are semantically meaningful and consistent with the underlying physical model.


```{code} python
:label: shape_semantics_example
:caption: Shape Semantics in PhlowerTensor

from phlower import phlower_tensor
import torch


# Example: PhlowerTensor representing pressure.
# Here, we assume that the number of time steps is 10, the number of spatial points is 100.
time_series_pressure = phlower_tensor(
    torch.rand(10, 100, 1),
    dimension={"M": 1, "T": -2, "L": -1},
    is_time_series=True,
)

print(time_series_pressure.is_time_series)
# Output: True

print(time_series_pressure.n_vertices())
# Output: 100

# Accessing the last time step of the time series
pressure_at_last_time_step = time_series_pressure[-1]

# Phlower automatically detects that accessed tensor is not a time series
print(pressure_at_last_time_step.is_time_series)
# Output: False

```

### Output of model structure

Phlower provides a feature to export the model structure in a [Mermaid](https://mermaid.js.org/) diagram format, making it easier to integrate them into documentation. [](#fig:sample_model) shows an example of model architecture diagram. This feature is useful for visualizing model structure and understanding the relationships between different components. 

:::{figure} images/sample_model.png
:label: fig:sample_model
Example of a model structure diagram generated by Phlower.
:::



## Related Work

Several libraries and frameworks have been developed to support physical dimensions in scientific computing and machine learning. 

* **Pint** @pint : A Python library for handling physical quantities with units. It provides functionality to define physical quantities with units and supports conversions between different unit systems. 

* **Python Quantities** @python-quantities : A library that extends NumPy to support physical quantities with units. It allows for dimensional analysis and unit conversions.

Unlike these libraries, Phlower does not provide unit conversion or unit arithmetic features, focusing instead on ensuring that tensor operations adhere to dimensional consistency rules. 


## Limitations

* **Runtime Overhead**: Dimensional consistency is enforced at runtime, which introduces additional computational overhead. This may impact performance in scenarios where a large numbers of tensor operations are performed, particularly in real-time applications.

* **Partial Compatibility with PyTorch**: Although PhlowerTensor is designed to integrate with PyTorch, certain advanced features such as in-place operations or custom autograd functions may require manual adaptation or may not be fully supported.


## Conclusion and Future Work

In traditional numerical simulations of physical phenomena, physical dimensions play a critical role in ensuring the correctness of computations.  
To bring this advantage into deep learning, Phlower integrates physical dimensional information into PyTorch tensor objects.  
By enforcing dimensional consistency, Phlower enables users to avoid errors and improve the reliability of their deep learning models.  
We believe that Phlower will serve as a valuable tool for researchers and engineers working at the intersection of deep learning and physical simulation.

Future work includes extending the library to support more complex physical models and enhancing the YAML-based model definition system.

