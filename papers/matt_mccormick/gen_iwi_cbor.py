from itkwasm_image_io import imread, imwrite

image = imread('vm_head_frozenct.mha')
imwrite(image, 'vm_head_frozenct.iwi.cbor')
