---
title: Challenges and Implementations for ML Inference in High-energy Physics
abstract: |
  At CERN, machine learning models are developed and deployed for a range of applications, including data analysis, event reconstruction, and classification. These models must be not only highly sophisticated but also optimized for efficient inference. One critical application is in trigger systems—designed to identify and select interesting events from an immense stream of experimental data. Experiments like ATLAS and CMS generate data at rates approaching 100 TB/s, requiring triggers to rapidly filter out irrelevant events.

  With the advent of the upcoming High-Luminosity phase of the Large Hadron Collider at CERN, the number of collisions will rise, thereby the data generation rates will further increase significantly. This requires major work to make the architectures for modelling and analysing the involved physical phenomenon further efficient and optimized.

  To address the requirements of deploying machine learning models in such high-throughput environments, the ML4EP team at CERN has been developing SOFIE—a tool that translates machine learning models into standalone C++ code that is easy to integrate, highly customizable, and optimized for inference. SOFIE is further integrated within the ROOT project and therefore provides a Python API for easier usage within machine learning pipelines. In this work, we review the challenges of running machine learning in production at large-scale experiments like those at CERN and explore solutions such as SOFIE to enhance performance and reliability.

---

## Introduction

Machine Learning inference is becoming increasingly critical across various domains, particularly in HEP, where efficient model evaluation is essential for production workflows. Integrating inference seamlessly into existing software systems, such as reconstruction, simulation, and analysis software, requires support for evaluating models directly within C++ code,
beyond the typical Python-based ML environments. Furthermore, effective thread management is crucial for leveraging models in multi-threaded environments, ensuring optimal performance in large-scale data processing tasks. In many HEP applications, inference must be
performed at the event level, often requiring single-batch processing while maintaining both
computational speed and memory efficiency. Addressing these challenges is key to enabling
fast and resource-efficient inference of ML models within complex scientific workflows.

## Background
While TensorFlow[@tensorflow2015-whitepaper] and PyTorch[@10.5555/3454287.3455008] provide robust inference capabilities, their use in C++
environments presents several challenges. These frameworks are primarily designed around
their native model formats, limiting flexibility when integrating externally trained models.
Using TensorFlow within a C++ environment is particularly challenging, as its C++ API is
not trivial to use and introduces significant dependencies, making deployment more complex.
Additionally, TensorFlow’s thread management can be challenging to control, and its inference engine is often not optimized for specific use cases, such as single-event evaluation in
HEP workflows.  

PyTorch, on the other hand, offers the Torch C++ library (LibTorch), which provides a more
convenient interface for C++ integration. It is generally easier to install and requires fewer
dependencies compared to TensorFlow. However, full support for all PyTorch extensions is
not always available, particularly for specialized libraries such as PyTorch Geometric or PyTorch Cluster, which are commonly used for Graph Neural Networks. Furthermore, certain
issues arise when converting models from ONNX to the Torch format, limiting the flexibility
of model deployment. These constraints highlight the need for a lightweight and efficient
inference solution that seamlessly integrates into C++-based data processing pipelines while
maintaining high performance and flexibility.

The Open Neural Network Exchange (ONNX)[@bai2019] provides a standardized format for describing and sharing deep learning models, facilitating interoperability across different frameworks. However, ONNX cannot fully represent all model architectures, particularly those
used in Graph Neural Networks. To enable the efficient inference of ONNX models, Microsoft developed ONNX Runtime[@onnxruntime], an open-source inference engine that supports both
C++ and Python environments. It offers flexibility by running on both CPUs and GPUs, with
NVIDIA GPU acceleration via TensorRT[5] and AMD support through ROCm[6].
ONNX Runtime has already been successfully integrated into HEP software frameworks, including ATLAS and CMS, where its convenient C++ API and fine-grained thread control
have proven valuable. As it is based on the ONNX format, trained models from TensorFlow
and PyTorch can be converted for use with ONNX Runtime. However, not all models are fully
compatible with ONNX, posing limitations when working with certain architectures. Despite
these constraints, ONNX and ONNX Runtime provide a promising solution for deploying
machine learning models efficiently within C++-based scientific computing workflows.

### AthenaTriton
AthenaTriton is a machine learning inference tool developed for the ATLAS experiment to enable scalable inference within the Athena software framework by leveraging the NVIDIA Triton Inference Server. It implements the Inference as a Service IaaS model, where Athena acts as a client that sends requests to a local or remote Triton server for executing ML models. This architecture supports dynamic offloading of computation to GPUs and improves overall resource utilization and load balancing. AthenaTriton uses the AthInferenceTritonTool, which conforms to the common IAthInferenceTool interface, and requires only the model name and server URL to operate. It converts input tensors to raw uint8_t data, sends them via gRPC, and returns the inference outputs in user-defined formats. Triton supports multiple ML backends including ONNX Runtime, TensorRT, TensorFlow, and PyTorch, as well as custom Python and C++ backends, enabling flexible deployment. The tool has been demonstrated using a GNN-based track-finding pipeline called GNN4ITk as a Service, which processes ATLAS ITk simulated data and shows efficient scaling on NVIDIA A100 GPUs. Performance evaluations using perf_analyzer revealed that with increasing concurrent model instances, throughput scaling efficiency remained above 98% while GPU utilization approached 45%. End-to-end tests with Athena clients showed that three concurrent threads achieved a 2.4× speedup, with strong scaling efficiency maintained. AthenaTriton thus provides a robust and maintainable solution for integrating ML inference into both online and offline ATLAS workflows.

### SONIC
The Service for Optimized Network Inference on Coprocessors is a framework adopted by the CMS and ATLAS experiments to facilitate heterogeneous computing through an IaaS. Rather than coupling CPUs directly to coprocessors, SONIC allows for more flexible deployment of ML and non-ML algorithms across specialized hardware such as GPUs, FPGAs, IPUs, and TPUs. This is especially relevant for the HL-LHC phase, where increasing data complexity demands efficient use of computing resources. In CMS, SONIC has been integrated into the MiniAOD data processing workflow using NVIDIA Triton servers to run inference tasks on GPUs, leveraging backends such as ONNX, TensorFlow, PyTorch, and Scikit-learn. Per-model optimization is achieved via Triton’s model analyzer tool, and large-scale tests have been conducted to evaluate performance under simultaneous client loads, with fallback CPU inference servers ensuring robust reliability. Additionally, SONIC supports non-ML acceleration; for example, CMS has implemented Patatrack-AAS and is developing Line Segment Tracking AAS for HL-LHC, both of which enable GPU-based tracking. In ATLAS, SONIC is being developed with the ACTS toolkit for charged particle tracking, including efforts like trac-cc to rewrite ACTS algorithms for GPU execution. The framework involves considerations for backend tuning, batch size optimization, load balancing via Kubernetes, and deployment across multiple sites, all aimed at enhancing scalability, latency performance, and coprocessor saturation in real production environments.

### E2E DL Inference framework in CMSSW
The End-to-End deep learning inference framework developed within the CMS Software Framework (CMSSW) enables direct use of low-level detector data for machine learning-based object and event classification. Unlike traditional workflows that rely on the particle flow algorithm, which may involve information loss due to simplification, this framework performs inference directly on raw detector inputs, supporting applications such as single-particle classification (electron, photon), jet tagging (quark, gluon, boosted top, tau), and exotic event reconstruction (e.g., H→AA→4γ). The system is structured around three main CMSSW packages—DataFormats, FrameProducer, and Taggers, and operates on detector outputs by generating cropped data frames centered on seed coordinates. These frames serve as inputs to ONNX-converted convolutional neural networks, integrated into the framework using the ONNX C++ API with GPU support. Benchmark studies demonstrate improved latency and throughput when using GPUs compared to CPUs, with event-level inference time reductions of up to 20% for E/Gamma and ~19% for jet-based classifiers. Inference throughput increased by 11–18% across taggers on Fermilab LPC GPUs, with additional scaling tested on NERSC Perlmutter systems. The E2E system is capable of performing stable, reproducible inference with support for future optimization, offering a scalable and high-performance solution for machine learning applications in CMS.

### hls4ml
hls4ml is an open-source software-hardware codesign workflow designed to bridge the gap between machine learning development and efficient hardware deployment on FPGAs and ASICs. It provides a seamless translation of trained neural networks—developed in standard ML frameworks such as TensorFlow, PyTorch, and QKeras—into HLS implementations suitable for resource-constrained, low-power, or low-latency environments. By supporting optimizations like quantization-aware training, pruning, configurable parallelization, and sparse matrix operations, hls4ml enables the generation of custom hardware accelerators that are tailored to both performance and energy constraints. The framework offers a Python-based API that allows for introspection, bit-accurate emulation, and visualization of the model’s architecture and numerical properties, making it accessible to domain scientists without deep hardware expertise. Its modular structure supports multiple backend tools and vendor-specific workflows, including Xilinx Vivado HLS, Intel Quartus HLS, and ASIC toolchains via Mentor Catapult HLS. As such, hls4ml facilitates rapid prototyping, exploration of design trade-offs, and deployment of machine learning models on edge devices in scientific and embedded computing contexts.

## SOFIE
To address the challenges of efficient machine learning inference in C++ environments, we
introduce SOFIE[7]—a tool within ROOT/TMVA[8] designed to generate optimized C++
code from trained ML models. SOFIE is capable of converting models in ONNX format to
its own Intermediate Representation (Fig. 1). Additionally, it provides limited support for
TensorFlow/Keras and PyTorch models, as well as message-passing Graph Neural Networks
from DeepMind’s Graph Nets library.
The key advantage of SOFIE is its ability to produce standalone C++ code that can be directly invoked within C++ applications with minimal dependencies—requiring only BLAS
for numerical computations. This makes integration seamless for high-energy physics workflows and other computationally demanding applications. Moreover, the generated code can
be compiled at runtime using ROOT[9] Cling Just-In-Time compilation, allowing for flexible
execution, including within Python environments. By eliminating the need for heavyweight
machine learning frameworks during inference, SOFIE offers a highly efficient and easily
deployable solution for ML model evaluation.

### Benchmarking results
The benchmarking study on SOFIE demonstrates its strong performance in event-level inference tasks common in HEP applications. Conducted on a standard Linux desktop equipped with an AMD Ryzen processor (24 threads, 4.4 GHz) and an NVIDIA RTX 4090 GPU, the benchmarks were run in single-thread mode to ensure a consistent evaluation of inference efficiency across frameworks. SOFIE was tested using both OpenBLAS and Intel MKL for CPU-based linear algebra, and leveraged SYCL with Intel oneAPI and PortBLAS for GPU acceleration. The results show that SOFIE consistently outperforms established frameworks such as ONNX Runtime and LibTorch in scenarios involving small to medium-sized models, particularly for linear architectures and VAEs, achieving lower latency and reduced memory consumption during single-event inference. In these cases, SOFIE benefits from its efficient code generation and use of high-performance mathematical libraries. However, for convolution-heavy models such as Conv2D, Conv3D, and ResNet, SOFIE lags behind due to the absence of optimized convolutional kernels, where LibTorch and ONNX Runtime perform significantly better. For GNNs, SOFIE shows favorable performance on small-scale inputs and maintains competitive scaling behavior, although its efficiency and memory usage degrade with increasing graph size due to the lack of dedicated optimizations. Overall, SOFIE provides a compelling inference solution for C++-based HEP workflows, with minimal dependencies and strong performance on key tasks, and ongoing development efforts aim to further enhance its GPU capabilities, memory optimizations, and support for complex model architectures.

## Optimization methods
Considering the benchmarking results of SOFIE, we have been working on adding optimization methods in SOFIE, to particularly improve the memory usage of the generated code during inference runtime for very large models. One important aspect of this have been on efficient memory reuse for intermediate tensors needed during the execution. Additionally, fusing machine learning operators further reduces the intermediate memory required by avoiding allocating memory for operations that can be performed in-place. While initially the generated code by SOFIE expanded tensors along dimensions for operations that had compatible but unequal shapes, efficient broadcasting mechanisms were developed to reduce further memory copy. 

### Memory reuse
In addition to weights, the execution of machine learning models require several intermediate tensors that handle and carry the results of the intermediate steps of a model execution. Therefore, for a model containing 2 set of layers of Dense and ReLU operations, we will need intermediate memory that shall collect the output from those layers and is supplied as inputs to the next, so a total of 3 intermediate tensors are required.

:::{figure} image_memory_reuse_1.png
:label: fig:memory_reuse_1
A model containing 2 sets of Dense and ReLU layers will need a total of 3 intermediate tensors
:::

Without any optimizations, SOFIE used to allocate memory for each of them, which is unnecessary if some of those tensors will not be required anymore if the execution flow has already progressed. In the example below, since Intermediate Tensor 1 will no longer be required when the execution flow is with the last Gemm Operator, therefore the Intermediate Tensor 3 can safely take its place in the memory instead of allocating additional memory.

:::{figure} image_memory_reuse_2.png
:label: fig:memory_reuse_2
In this example, we assume the intermediate tensors are of Float data type and are of length 2 each.
:::

Therefore, we developed a memory reuse mechanism that evaluates memory required and possible positioning ahead of time, which is then used to allocate a big block of memory that can be used to develop pointers to assign memory blocks for the intermediate tensors.
:::{prf:algorithm} Memory reuse mechanism
:label: Memory reuse mechanism

**Inputs** Given a list of operators with input and output tensors, a Total Stack to track the total memory occupied, an Available Stack to track the vacant positions in Total Stack

**Output** Compute positions to assign for intermediate tensors.

1. For each operator:
  1. For every output tensor:  
    1. Check if Available Stack has any suitable memory chunk  
    2. If yes, reposition it for reuse  
    3. If no, obtain new memory from pool and track it in Total Stack  
  
  2. For every input tensor:  
    1. Check if it is the last operator which is using this as input  
    2. If yes, consider it in available memory (for memory reuse)  
      1. Check if the newly available chunk can be coalesced with an adjoining chunk to make a larger block
:::

### Multi-Operator Fusion
Several operators in a machine learning model involve weight-less in-place operations, i.e. operatations are element-wise are not dependent on the overall structure of the input tensor, additionally the shapes of their input and output tensors remain unchanged. Such operations include ReLU, BatchNormalization, LayerNormalization, etc., can be fused together so as to avoid allocating memory for their outputs, instead computations can be performed directly on their input tensors which are then dispatched as output for the next operator's utilization.

```{admonition} Operator Fusion Algorithm
:class: tip

The following algorithm describes the logic for operator fusion in a computational graph.

1. **For each operator** in the computation graph:
   1. **Check if it is an anchor operation** (e.g., `GEMM`, `Conv`, etc.):
      1. If **yes**:
         1. **Check if the next operator is fusable**, i.e., an in-place, weight-less operation:
            1. If **yes**:
               1. Fuse it with the preceding operation.
               2. **Check if this is the last fusable operation** in the chain:
                  1. If **yes**:
                     1. Break the fusion chain and resume the mechanism from the next operator.
                  2. If **no**:
                     1. Continue to the next operator.
            2. If fusion is **not possible**:
               1. Move to the next operator.
      2. If it is **not an anchor operation**:
         1. Fusion is not possible, move to the next operator.
```
### Efficient Broadcasting
Operations in the execution of a machine learning model require tensor computations that may involve binary operations, where if the input tensors are of not the same shape, they are broadcasted to a matching shape before being computed. Broadcasting is a process of of extending towards a dimension in a tensor without actually being expanding through copying. While initially the generated code by SOFIE used to expand through un-equal dimensions, a more efficient broadcasting algorithm was developed to avoid further memory allocations and movements.

## Inference on heterogeneous architectures
We have been working on adding the support in SOFIE for inference on heterogeneous architectures by abstracting device-specific operations through a flexible buffer-accessor model using libraries like SYCL and ALPAKA. At the core of its design is an Infer function that allows users to explicitly control execution on Intel, NVIDIA, or AMD GPUs. During session instantiation, SOFIE initializes input and output buffers, enabling seamless buffer-based data exchange. The backend automatically selects appropriate BLAS routines based on the target execution architecture, ensuring optimal performance without manual intervention. Support for inference on heterogeneous architectures is further enhanced by the use of abstraction libraries such as SYCL and ALPAKA, which offer platform-agnostic interfaces for memory management, making them well-suited for inference code generation in SOFIE. Development is ongoing, with prototypes for SYCL and ALPAKA integration already underway. A CUDA-based prototype supporting GEMM and ReLU operations is currently under testing, while support for Intel and AMD GPUs is planned.

:::{figure} image_heterogeneous.png
:label: fig:heterogeneous
SOFIE supporting inference of machine learning models on heterogeneous architectures through abstract libraries.
:::

## Experimental Evaluation
Experimentations were performed on similar initial benchmarking conditions to evaluate the optimization methods developed on the ParticleNet model developed and utilized by the CMS Experiment of CERN. We observed an approximate 50% reduction in the memory usage from SOFIE's previous iteration, but it was observed that SOFIE still requires more memory for its operation in comparison to ONNXRuntime, although SOFIE's inference was time was improved lowering its latency lower than that of ONNXRuntime.

:::{figure} image_eval_optimization.png
:label: fig:evaluation_optim
Evaluations shows SOFIE's improved memory usage but still requires more memory than ONNXRuntime
:::

## Acknowledgement
This work has been funded by the Eric & Wendy Schmidt Fund for Strategic Innovation through the CERN Next Generation Triggers project under grant agreement number SIF-2023-004.

