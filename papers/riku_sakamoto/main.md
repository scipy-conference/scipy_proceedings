---
# Ensure that this title is the same as the one in `myst.yml`
title: "Phlower: A Deep Learning Framework Supporting PyTorch Tensors with Physical Dimensions"
abstract: |
  Physics-informed Machine Learning, which integrates machine and deep learning with physical simulations, is gaining attention as a powerful tool for modeling physical phenomena. In physical simulations, we must carefully handle physical dimensions — such as time (T), mass (M), and length (L) — to ensure reliable calculations. Therefore, incorporating physical dimensions into tensors is essential. 
  To address this challenge, we created [Phlower](https://github.com/ricosjp/phlower), an open-source deep learning library that extends PyTorch tensors to support physical dimensions and enforces dimensional consistency throughout computations. In this talk, we will present Phlower’s core features and demonstrate how it ensures dimensional correctness in deep learning tasks.

---

## Introduction

The simulation of physical phenomena is essential in our daily lives for tasks such as designing vehicles and forecasting weather. As a new paradigm, Physics-informed Machine Learning, which integrates machine and deep learning with physical simulations, is gaining attention as a powerful tool, particularly with the evolution of Physics-Informed Neural Networks (PINNs) @RAISSI2019686 and Graph Neural Networks (GNNs). Compared to existing simulation methods, it is expected to predict physical phenomena faster or uncover new laws that may have never been detected before.

When performing numerical simulations of physical phenomena, we must carefully handle physical dimensions — such as time (T), mass (M), and length (L) — to ensure the correctness of computations. To apply this concept to deep learning, handling physical dimensions offers two benefits. First, maintaining consistency of physical dimensions enhances the reliability of the model architecture. Second, physical dimensions themselves can be useful for scaling the inputs to machine learning models.

To address these needs, we developed Phlower, an open-source deep learning framework that allows PyTorch @NEURIPS2019_bdbca288 tensor objects to carry physical dimensions. In this presentation, we will introduce Phlower’s key features and demonstrate how physical dimensions are handled in deep learning workflows.

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

print(kinetic_energy.dimesion)
# Output: PhlowerDimensionTensor(T: -2.0, L: 2.0, M: 1.0, I: 0.0, Theta: 0.0, N: 0.0, J: 0.0)
```


The rest of this paper is composed as follows:

* In Section 2, we desribe the core concept of Phlower and its key features. Also, we introduce the basic usage of `PhlowerTensor`, which is a wrapper class of PyTorch @NEURIPS2019_bdbca288 Tensor object and enables physical dimension tracking.

* In Section 3, we explain the use cases of Phlower. Two examples are demonstrated using PhlowerTensor: a simple regression of physical variables and the implementation of Physics-Informed Neural Networks (PINNs).

* In Section 4, we discuss the additional features of Phlower, including YAML-based model definition and shape positioning.

* In Section 5, we compare Phlower with related work and discuss its limitations.


## Design and Core Concepts


### Motivation

PyTorch is a widely used deep learning framework that provides powerful tensor operations and automatic differentiation capabilities. However, it does not inherently support physical dimensions, which are crucial for ensuring the correctness of computations in physics-related applications. This limitation can lead to errors and inconsistencies when performing operations on tensors representing physical quantities.Here, physical dimensions refer to the fundamental quantities that describe physical phenomena, such as time (T), mass (M), length (L), electric current (I), temperature (Theta), amount of substance (N), and luminous intensity (J). 

Therfore, Phlower introduces `PhlowerTensor` which is a wrapper class of PyTorch Tensor object and enables physical dimension tracking. This ensures that tensor operations adhere to dimensional consistency rules. Thanks to its lightweight implementation, converting between torch.Tensor and PhlowerTensor is straightforward. Moreover, PhlowerTensor maintains compatibility with PyTorch’s autograd system and core tensor operations like `sum()`, making it easy to integrate into existing PyTorch-based applications with minimal code modifications.



### Basic Usage

In this section, we will introduce the basic usage of `PhlowerTensor` and how it integrates with PyTorch.

#### Defining PhlowerTensor

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

PhlowerTensor enforces dimensional consistency during tensor operations. 

In the following example, we will demonstrate how PhlowerTensor ensures dimensional consistency when computing kinetic energy, which is defined as {math}` E = 0.5 \cdot m \cdot \bm{v}^2 `, where {math}` m ` is mass and {math}` \bm{v} ` is velocity. Here, we assume that there are 10 observing points, the velocity is a 3-dimensional vector (e.g., in 3D space), and the mass is a scalar value. The resulting kinetic energy will have dimensions {math}` M^1 L^2 T^{-2} `, which corresponds to the physical dimension of energy.


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

print(kinetic_energy.dimesion)
# Output: PhlowerDimensionTensor(T: -2.0, L: 2.0, M: 1.0, I: 0.0, Theta: 0.0, N: 0.0, J: 0.0)

```


On the other hand, if we attempt to perform an operation with incompatible dimensions, such as adding a pressure tensor to a velocity tensor, PhlowerTensor will raise an error:

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

#### Array operation

In this section, we will demonstrate how PhlowerTensor supports array operations while maintaining dimensional consistency.

stacking tensors with compatible dimensions is straightforward. For example, we can stack multiple velocity tensors along a new dimension.
You can find that the resulting tensor is also PhlowerTensor which has dimension although `torch.stack` is called.

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

Here, we will demonstrate two use cases which can be benefitail to use PhlowerTensor: the implementation of Physics-Informed Neural Networks (PINNs).


### Preventing Dimensional Inconsistencies in Physics-Informed Models

In physics-informed neural networks (PINNs), the loss function often includes terms derived from differential equations. For example, when modeling a system governed by Navier-Stokes equations, the loss function is composed of terms like the continuity equation and momentum equations. These equations involve derivatives with respect to spatial and temporal dimensions, which can lead to dimensional inconsistencies if not handled correctly.

### Symbolic Unit Tracking in Scientific Machine Learning

When handling physical measurements such as temperature, velocity, and pressure, embedding unit information directly into the tensors allows machine learning models to benefit from automatic feature validation. For instance, the model can reject invalid combinations (e.g., subtracting pressure from velocity) and suggest unit-consistent preprocessing steps. This feature is particularly valuable in domains where data is collected from heterogeneous sources with different measurement conventions.



## Additional Features

### Yaml-based definition

Phlower offers a YAML-based model definition system that allows users to define, configure, and reuse machine learning models efficiently. This simplifies experimentation by enabling users to modify model architectures and hyperparameters quickly without altering code.

```{code} yaml
:label: phlower_yaml_example
:caption: Example of YAML-based model definition in Phlower (Extract from `phlower/examples/concat_mlp.yaml`)

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


### Shape Positioning

the index of PhlowerTensor's shape has a meaningful position. For example, if a tensor represents a velocity field in 3D space, the first dimension might represent time, the second dimension might represent the spatial x-coordinate, and the third dimension might represent the spatial y-coordinate. This positional information is crucial for ensuring that operations on tensors are semantically meaningful and consistent with the underlying physical model.

```{code} python
:label: shape_positioning_example
:caption: Shape positioning in PhlowerTensor

from phlower import phlower_tensor
import torch


# Example: PhlowerTensor representing pressure.
# Here, we assume that the number of time steps is 10, the number of spatial points is 100, 
#  and the number of points is 100
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

Phlower provides a feature to output the model structure in a mermaid diagram format. This feature is useful for visualizing the model architecture and understanding the relationships between different components of the model. The generated diagram can be easily integrated into documentation or presentations, making it easier to communicate the model design to others.

:::{figure} images/sample_model.png
:label: fig:sample_model
Example of a model structure diagram generated by Phlower.
:::



## Related Work

There are several libraries and frameworks that address the need for physical dimensions in scientific computing and machine learning. Some notable ones include:
* **Pint** @pint : A Python library for handling physical quantities with units. It provides a way to define and manipulate physical dimensions, but it does not integrate directly with PyTorch tensors.
* **Python Quantities** @python-quantities : A library that extends NumPy to support physical quantities with units. It allows for dimensional analysis and unit conversions, but it is not specifically designed for deep learning applications.



## Limitation

* **Runtime Overhead**: Dimensional consistency is enforced at runtime, which introduces additional computational overhead. This may impact performance in scenarios where large numbers of tensor operations are performed, especially in real-time applications.

* **Partial Compatibility with PyTorch**: Although PhlowerTensor is designed to integrate with PyTorch, certain advanced features such as in-place operations or custom autograd functions may require additional wrapping or are not fully supported.

## Conclusion and Future Work

In existing numerical simulations of physical phenomena, physical dimensions play an important role in ensuring the correctness of calculations. To apply the advantages to deep learning, Phlower integrates physical dimensional information into the Tensor object. Thanks to this feature, users can avoid dimensional inconsistency and improve the reliability of their deep learning models. We believe that Phlower will be a valuable tool for researchers and engineers working at the intersection of deep learning and physical simulation. 

Future work includes expanding the library to support more complex physical models and enhancing the YAML-based model definition system.

