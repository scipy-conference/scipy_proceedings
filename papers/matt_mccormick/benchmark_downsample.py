from time import perf_counter

import numpy as np
from rich import print
import matplotlib.pyplot as plt

import itkwasm_image_io
import itkwasm_downsample
import itkwasm_downsample_wasi
import itkwasm_downsample_cucim
from itkwasm_downsample import downsample_sigma
import itk
import itkwasm

image_filename = 'vm_head_frozenct.mha'
image = itkwasm_image_io.imread(image_filename)
# copy
image_cupy = itkwasm.cast_image(image)
image_cupy.data = itkwasm.array_like_to_cupy_array(image_cupy.data)
image_native = itk.imread(image_filename)
shrink_factors = [4, 4, 2]
iterations=10

def measure_fn(fn, iterations=3):
    # skip import, loading side effects
    fn()
    timings = []
    for _ in range(iterations):
        start_time = perf_counter()
        fn()
        end_time = perf_counter()
        timings.append(end_time - start_time)

    print(timings)
    mean = np.mean(timings)
    std = np.std(timings)

    print('mean, std: ', mean, std)
    return mean, std

def downsample_itk_native(shrink_factors, n_threads=1):
    sigma_values = downsample_sigma(shrink_factors)

    gaussian_filter = itk.DiscreteGaussianImageFilter.New(image_native)
    gaussian_filter.SetSigmaArray(sigma_values)
    gaussian_filter.SetUseImageSpacingOff()

    mt = gaussian_filter.GetMultiThreader()
    mt.SetGlobalDefaultNumberOfThreads(n_threads)
    mt.SetMaximumNumberOfThreads(n_threads)

    input_origin = itk.origin(image_native)
    input_spacing = itk.spacing(image_native)
    input_size = itk.size(image_native)

    dimension = image_native.GetImageDimension()
    output_origin = itk.Point[itk.D, dimension]()
    output_spacing = itk.Vector[itk.D, dimension]()
    output_size = itk.Size[dimension]()
    for i in range(dimension):
        crop_radius_value = 0.0
        output_origin[i] = input_origin[i] + crop_radius_value * input_spacing[i]
        output_spacing[i] = input_spacing[i] * shrink_factors[i]
        output_size[i] = int(max(0, (input_size[i] - 2*crop_radius_value) // shrink_factors[i]))

    interpolator = itk.LinearInterpolateImageFunction.New(image_native)
    shrink_filter = itk.ResampleImageFilter.New(gaussian_filter)
    mt = shrink_filter.GetMultiThreader()
    mt.SetGlobalDefaultNumberOfThreads(n_threads)
    mt.SetMaximumNumberOfThreads(n_threads)
    shrink_filter.SetInterpolator(interpolator)
    shrink_filter.SetOutputOrigin(output_origin)
    shrink_filter.SetOutputSpacing(output_spacing)
    shrink_filter.SetSize(output_size)
    shrink_filter.SetOutputDirection(image_native.GetDirection())
    shrink_filter.SetOutputStartIndex(image_native.GetLargestPossibleRegion().GetIndex())
    shrink_filter.UpdateLargestPossibleRegion()

def downsample_itk_wasm():
    downsampled = itkwasm_downsample.downsample(image, shrink_factors=shrink_factors)

def downsample_itk_wasm_wasi():
    downsampled = itkwasm_downsample_wasi.downsample(image, shrink_factors=shrink_factors)

def downsample_itk_wasm_cucim():
    downsampled = itkwasm_downsample_cucim.downsample(image_cupy, shrink_factors=shrink_factors)

print('itk-wasm')
itk_wasm_mean, itk_wasm_std = measure_fn(downsample_itk_wasm, iterations)
print('itk-wasm-wasi')
itk_wasm_wasi_mean, itk_wasm_wasi_std = measure_fn(downsample_itk_wasm_wasi, iterations)
print('itk-wasm-cucim')
itk_wasm_cucim_mean, itk_wasm_cucim_std = measure_fn(downsample_itk_wasm_cucim, iterations)
print('itk-native')
itk_native_mean, itk_native_std = measure_fn(lambda: downsample_itk_native(shrink_factors), iterations)
print('itk-native 10 threads')
itk_native_mean_10_threads, itk_native_std_10_threads = measure_fn(lambda: downsample_itk_native(shrink_factors, 10), iterations)

data = [(itk_wasm_wasi_mean, itk_wasm_wasi_std), (itk_native_mean, itk_native_std), (itk_native_mean_10_threads, itk_native_std_10_threads), (itk_wasm_cucim_mean, itk_wasm_cucim_std)]
means = [d[0] for d in data]
stddevs = [d[1] for d in data]
labels = ['ITK-Wasm WASI,\n1 thread', 'Native ITK Python,\n1 thread', 'Native ITK Python,\n10 threads', 'ITK-Wasm CuCIM']

x = np.arange(len(data))

plt.figure(figsize=(10, 6))
plt.bar(x, means, yerr=stddevs, capsize=5, color='skyblue', alpha=0.7)
# plt.xlabel('Benchmark')
plt.ylabel('Performance (sec) - lower is better')
plt.title('Downsample Performance')
plt.xticks(x, labels)

plt.savefig('figures/benchmark_results.png', format='png')
plt.savefig('figures/benchmark_results.svg', format='svg')

plt.show()