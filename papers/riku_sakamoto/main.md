---
# Ensure that this title is the same as the one in `myst.yml`
title: "Phlower: A Deep Learning Framework Supporting PyTorch Tensors with Physical Dimensions"
abstract: |
  We present [Phlower](https://github.com/ricosjp/phlower), an open-source deep learning library that extends PyTorch tensors to incorporate physical dimensions — such as time ( _T_ ), mass ( _M_ ), and length ( _L_ ) — and enforces dimensional consistency throughout computations. 
  
  Deep learning models applied to physical phenomena often involve operations on tensors representing physical quantities, where maintaining dimensional correctness is essential. To address this need, Phlower introduces `PhlowerTensor`, a wrapper class around PyTorch tensors that tracks physical dimensions and ensures that tensor operations comply with dimensional consistency rules.
  
  This paper introduces the core features of Phlower and demonstrates its capability to maintain dimensional correctness through representative examples.

---

## Introduction

The simulation of physical phenomena plays a critical role in various aspects of daily life, including vehicle design and weather forecasting. As a new paradigm, _physics-informed machine learning_, which integrates machine learning with physical simulations, has recently gained attention, particularly with the emergence of Physics-Informed Neural Networks (PINNs) [@RAISSI2019686] and Graph Neural Networks (GNNs). Compared to traditional simulation methods, it offers the potential to accelerate predictions of physical behavior and to uncover new physical laws that might otherwise remain hidden.

In numerical simulations of physical phenomena, it is essential to handle physical dimensions to ensure the correctness of computations. As shown in [](#physical_dimension), Physical dimensions are the fundamental quantities used to describe physical systems. These include time ( {math}`T` ), mass ( {math}`M` ), length ( {math}`L` ), electric current ( {math}`I` ), thermodynamic temperature ( {math}`\Theta` ), amount of substance ( {math}`N` ), and luminous intensity ( {math}`J` ). Extending this concept to deep learning provides several advantages. For example, enforcing dimensional consistency enhances the reliability of model architectures by preventing invalid operations between physically incompatible quantities. This capability helps avoid subtle implementation errors and improves code readability and maintainability, especially in scientific and engineering applications where preserving physical meaning is critical. 

To address these needs, we developed Phlower, an open-source deep learning framework that enables PyTorch [@NEURIPS2019_bdbca288] tensor objects to carry physical dimensions. An example illustrating its usage is shown in the [](#example_phlower_tensor). 

PyTorch is a widely used deep learning framework that provides powerful tensor operations and automatic differentiation capabilities. However, it does not natively support physical dimensions. To fill this gap, Phlower introduces `PhlowerTensor` which is a wrapper class of PyTorch Tensor object and enables physical dimension tracking. This ensures that tensor operations adhere to dimensional consistency rules. Thanks to its lightweight implementation, converting between PyTorch's tensor object and `PhlowerTensor` is straightforward. Moreover, PhlowerTensor maintains compatibility with PyTorch’s autograd system and core tensor operations such as `sum()`, enabling seamless integration into existing PyTorch-based workflows with minimal code modifications.

:::{table} Physical Dimensions [@thompson2008use]
:label: physical_dimension

| quantity  | Symbol |
| --------- | --- |
| time | {math}`T` |
| mass | {math}`M` |
| length | {math}`L` |
| electric current | {math}`I` |
| thermodynamic temperature | {math}`\Theta` |
| amount of substance | {math}`N` |
| luminous intensity | {math}`J` |

:::


```{code} python
:label: example_phlower_tensor
:caption: PhlowerTensor carrying physical dimensions

from phlower import phlower_tensor
import torch

# Example: Calculating kinetic energy (E = 0.5 * m * v^2)

# A mass (m) has dimensions M^1
mass = phlower_tensor(torch.rand(10, 1), dimension={"M": 1})

# A velocity (v) has dimensions L^1 T^-1
velocity = phlower_tensor(torch.rand(10, 3, 1), dimension={"L": 1, "T": -1})

kinetic_energy = 0.5 * mass * torch.sum(velocity ** 2, axis=-1)

print(kinetic_energy.dimension)
# Output: PhlowerDimensionTensor(T: -2.0, L: 2.0, M: 1.0, I: 0.0, Theta: 0.0, N: 0.0, J: 0.0)
```

In this paper, we will introduce Phlower’s key features and demonstrate how physical dimensions are handled in deep learning workflows. The remainder of this paper is organized as follows:

* Section 2 introduces the core concept of Phlower and its key features. Defining `PhlowerTensor` and enforcing dimensional consistency are discussed, along with examples of tensor operations that maintain physical dimensions.

* Section 3 presents possible use cases of PhlowerTensor in deep learning applications, particularly in scientific machine learning and physics-informed models. It highlights how PhlowerTensor can prevent dimensional inconsistencies and enforce dimensional validity in feature engineering.

* Section 4 discusses additional features of Phlower, including YAML-based model definition and shape semantics. These features enhance the usability and flexibility of Phlower in deep learning workflows.

* Section 5 compares Phlower with related work, highlighting its unique features and advantages over existing libraries for handling physical dimensions in scientific computing and machine learning.


## Core Concepts

### Design Philosophy

Phlower is designed to extend PyTorch tensors with support for physical dimensions while maintaining full compatibility with existing PyTorch workflows.  
Its design is guided by the following key principles:

* **Dimensional Consistency**: Phlower enforces dimensional consistency during tensor operations, ensuring that operations involving tensors with incompatible physical dimensions raise informative errors. This prevents subtle bugs and improves the reliability of deep learning models.
* **Seamless Integration**: Phlower is designed to integrate seamlessly with PyTorch, allowing users to leverage existing PyTorch functionalities while adding physical dimension tracking. This includes compatibility with PyTorch's autograd system and core tensor operations.
* **Lightweight Implementation**: Phlower is implemented as a lightweight wrapper around PyTorch tensors, minimizing the overhead of using physical dimensions. This allows users to easily convert between PyTorch tensors and `PhlowerTensor` without significant performance impact.


### Defining PhlowerTensor

[](#create_phlower_tensor) shows how to create a `PhlowerTensor` from a PyTorch tensor. The `phlower_tensor` function takes a PyTorch tensor and a dictionary specifying the physical dimensions. 


```{code} python
:label: create_phlower_tensor
:caption: Create PhlowerTensor from PyTorch Tensor

from phlower import phlower_tensor
import torch

# Create a PhlowerTensor with physical dimensions at each point
# Example: A tensor representing velocity with dimensions L^1 T^-1
velocity = phlower_tensor(torch.rand(10, 3), dimension={"L": 1, "T": -1})
print(velocity)

# Example: A tensor representing pressure with dimensions M^1 T^-2 L^-1
pressure = phlower_tensor(torch.rand(10, 1), dimension={"M": 1, "T": -2, "L": -1})
print(pressure)

```


### Dimensional Consistency

PhlowerTensor enforces dimensional consistency during tensor operations. [](#kinetic_energy_example) demonstrates how PhlowerTensor ensures dimensional correctness when computing kinetic energy, defined as {math}` E = 0.5 \cdot m \cdot \mathbf{v}^2 `, where {math}` m ` is mass and {math}` \mathbf{v} ` is velocity. 

As illustrated in [](#fig:fig_points), this example assumes that each point is associated with a three-dimensional velocity vector and a scalar mass.  
The resulting kinetic energy has dimensions {math}` M^1 L^2 T^{-2} `, corresponding to the physical dimension of energy.


:::{figure} images/figure1.png
:label: fig:fig_points
:width: 400px
Example of point clouds. Each point has mass and velocity.
:::


```{code} python
:label: kinetic_energy_example
:caption: Calculating kinetic energy with PhlowerTensor

from phlower import phlower_tensor
import torch

# Example: Calculating kinetic energy (E = 0.5 * m * v^2)

# Mass has dimensions M^1
mass = phlower_tensor(torch.rand(10, 1), dimension={"M": 1})

# Velocity has dimensions L^1 T^-1
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
    # Output: Error: Add operation for different physical dimensions is not allowed.

```

### Array Operations

This section demonstrates how `PhlowerTensor` supports array operations while maintaining dimensional consistency.
For example, [](#stacking_tensors_example) shows that multiple velocity tensors can be stacked along a new dimension.
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
stacked_velocity = torch.stack([velocity1, velocity2])
print(stacked_velocity.shape)
# Output: torch.Size([2, 10, 3])
print(stacked_velocity.dimension)
# Output: PhlowerTensor with dimensions L^1 T^-1 and shape (2, 10, 3)

```


## Use Cases

This section presents potential use cases of PhlowerTensor in scientific machine learning and physics-informed models. PhlowerTensor's ability to track physical dimensions and enforce dimensional consistency makes it a valuable tool for ensuring the correctness of computations in these domains.


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

Each index in the shape of a `PhlowerTensor` carries semantic meaning, a concept referred to as _shape semantics_ in Phlower.  
[]( #shape_semantics_example ) illustrates how `PhlowerTensor` can represent time-series pressure using shape semantics.  
This positional information is crucial for ensuring that tensor operations remain semantically meaningful and consistent with the underlying physical model.


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

Several libraries and frameworks have been developed to support physical dimensions in scientific computing.

* **Pint** @pint : A Python library for handling physical quantities with units. It provides functionality to define physical quantities with units and supports conversions between different unit systems. 

* **Python Quantities** @python-quantities : A library that extends NumPy to support physical quantities with units. It allows for dimensional analysis and unit conversions.

Unlike these libraries, Phlower does not provide unit conversion or unit arithmetic features, focusing instead on ensuring that tensor operations adhere to dimensional consistency rules. 


## Limitations

* **Runtime Overhead**: Dimensional consistency is enforced at runtime, which introduces additional computational overhead. This may impact performance in scenarios where a large numbers of tensor operations are performed, particularly in real-time applications.

* **Partial Compatibility with PyTorch**: Although PhlowerTensor is designed to integrate with PyTorch, certain advanced features such as in-place operations or custom autograd functions may require manual adaptation or may not be fully supported.


## Conclusion and Future Work

In traditional numerical simulations of physical phenomena, physical dimensions play a critical role in ensuring the correctness of computations.  
To bring this advantage into deep learning, Phlower integrates physical dimensional information into PyTorch tensor objects. By enforcing dimensional consistency, Phlower enables users to avoid errors and improve the reliability of their deep learning models. We believe that Phlower will serve as a valuable tool for researchers and engineers working at the intersection of deep learning and physical simulation.

Future work includes extending the library to support more complex physical models and enhancing the YAML-based model definition system.
