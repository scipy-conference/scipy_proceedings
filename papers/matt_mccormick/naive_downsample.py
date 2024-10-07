#!/usr/bin/env python

from ngff_zarr import from_ngff_zarr, Multiscales, Metadata, to_ngff_zarr

multiscales = from_ngff_zarr('vm_head_frozenct.zarr')
metadata = multiscales.metadata

from rich import print

print('original', multiscales)

scale2_image = multiscales.images[2]

scale2_metadata = Metadata(axes=metadata.axes, datasets=[metadata.datasets[2]], name=metadata.name, version=metadata.version, coordinateTransformations=metadata.coordinateTransformations)
multiscales_antialias = Multiscales(images=[scale2_image], metadata=scale2_metadata)

print('scale2 antialiased', multiscales_antialias)

to_ngff_zarr('antialias.ome.zarr', multiscales_antialias)

scale0_image = multiscales.images[0]
scale2_image.data = scale0_image.data[::2, ::4, ::4]
scale2_image.data = scale2_image.data.rechunk(64)

multiscales_aliased = Multiscales(images=[scale2_image], metadata=scale2_metadata)

print('scale2 aliased', multiscales_aliased)

to_ngff_zarr('aliased.ome.zarr', multiscales_aliased)