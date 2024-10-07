from itkwasm_image_io import imread, imwrite
from itkwasm_downsample import downsample

image = imread('vm_head_frozenct.iwi.cbor')
downsampled = downsample(image, shrink_factors=[4, 4, 2])
imwrite(downsampled, 'downsampled.iwi.cbor')
