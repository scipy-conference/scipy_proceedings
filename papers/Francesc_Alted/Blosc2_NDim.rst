:author: Project Blosc
:email:
:institution: Project Blosc
:equal-contributor:
:bibliography: mybib

:author: Francesc Alted
:email: francesc@blosc.org
:institution: Project Blosc
:equal-contributor:
:corresponding:

:author: Marta Iborra
:email: martaiborra24@gmail.com
:institution: Project Blosc
:equal-contributor:

:author: Oscar Guiñón
:email: soscargm98@gmail.com
:institution: Project Blosc
:equal-contributor:

:author: David Ibáñez
:email: jdavid.ibp@gmail.com
:institution: Project Blosc

:author: Sergio Barrachina
:email: barrachi@uji.es
:institution: Universitat Jaume I


---------------------------------------------------------------------------------
Using Blosc2 NDim As A Fast Explorer Of The Milky Way (Or Any Other NDim Dataset)
---------------------------------------------------------------------------------

.. class:: abstract

    N-dimensional datasets are widely used in many scientific fields. Quickly accessing subsets of these large datasets is critical for an efficient exploration experience. To achieve this goal, we have added support for multidimensional datasets to Blosc2, a compression and format library. This extended library can effectively deal with n-dimensional large sparse datasets, as the zeroed parts are almost entirely suppressed, while the non-zero parts are stored in smaller sizes than their uncompressed counterparts. In addition to this, the new two level data partition used in Blosc2 reduces the need for decompressing unnecessary data, which allows top-class slicing speed.

    The Blosc2 NDim layer enables the creation and reading of n-dimensional datasets in an extremely efficient manner. This is due to a completely general n-dim 2-level partitioning, which allows for slicing and dicing of arbitrary large (and compressed) data in a more fine-grained way. Having a second partition provides a better flexibility to fit the different partitions at the different CPU cache levels, making compression even more efficient.

    As an example, we will demonstrate how Blosc2 NDim enables fast exploration of the Milky Way using the Gaia DR3 dataset. This catalog contains information on 1.7 billion stars in our galaxy, but we have chosen to include just the stars that are in a sphere of 10 thousand light-years radius (centered in the Gaia telescope), which accounts for 0.7 billion stars. The total size of the dataset of star positions is 7.3 TB, but when compressed, it is reduced to just 6.5 GB, making it easy to fit into the memory of modern computers for being processed.

.. class:: keywords

    explore datasets, n-dimensional datasets, Gaia DR3, Milky Way, Blosc2, compression

Introduction
------------

The exploration of n-dimensional datasets is a common practice in many areas of science. However, one of its drawbacks is that the explored datasets size can become very large, which will slow down the exploration process significantly. In this paper, we demonstrate how Blosc2 NDim can be used to accelerate the exploration of huge n-dimensional datasets.

Blosc is a high-performance compressor optimized for binary data. Its design enables faster transmission of data to the processor cache than the traditional, non-compressed, direct memory fetch approach using an OS call to ``memcpy()``. This can be helpful not only in reducing the size of large datasets on-disk and in-memory, but also in accelerating memory-bound computations, which are typical in big data processing.

Blosc uses the blocking technique :cite:`FA10-starving-cpus` to minimize activity on the memory bus. The technique divides datasets into blocks small enough to fit in the caches of modern processors, where compression/decompression is performed. Blosc also takes advantage of SIMD (SSE2, AVX2, NEON…) and multi-threading capabilities in modern multi-core processors to maximize the compression/decompression speed.

In addition, using the Blosc compressed data can accelerate memory-bound computations when enough cores are dedicated to the task. Figure :ref:`sum-precip` provides a real example of this.

.. figure:: sum_openmp-rainfall.png
   :scale: 40%

   Speed for summing up a vector of real float32 data (meteorological precipitation) using a variety of codecs provided by Blosc2. Note that the maximum speed is achieved when utilizing the maximum number of (logical) threads available on the computer (28), where different codecs are allowing faster computation than using uncompressed data. Benchmark performed on a Intel i9-10940X CPU, with 14 physical cores. More info at :cite:`BDT18-breaking-memory-walls`. :label:`sum-precip`


Blosc2 is the latest version of the Blosc 1.x series, which is used in many important libraries, such as HDF5 :cite:`hdf5`, Zarr :cite:`zarr`, and PyTables :cite:`pytables`. Its NDim feature excels at reading multi-dimensional slices, thanks to an innovative pineapple-style partitioning technique :cite:`BDT23-blosc2-ndim-intro`. This enables fast exploration of general n-dimensional datasets, including the 3D Gaia dataset.

The Gaia dataset
----------------

The Gaia DR3 dataset is a catalog containing information on 1.7 billion stars in our galaxy. For this work, we extracted the 3D coordinates of 1.4 billion stars (those with non-null parallax values). When stored as a binary table, the dataset is 22 GB in size (uncompressed).

However, we converted the tabular dataset into a sphere with a radius of 10,000 light years and framed it into a 3D array of shape (20,000, 20,000, 20,000). Each cell in the array represents a cube of 1 light year per side and contains the number of stars within it. Given that the average distance between stars in the Milky Way is about 5 light years, very few cells will contain more than one star (e.g. the maximum of stars in a single cell in our sphere is 6). This 3D array contains 0.7 billion stars, which is a significant portion of the Gaia catalog.

The number of stars is stored as a uint8, resulting in a total dataset size of 7.3 TB. However, compression can greatly reduce its size to 6.5 GB since the 3D array is very sparse. Blosc2 can compress the zeroed parts almost entirely.

In addition, we store other data about the stars in a separate table indexed with the position of each star (using PyTables). For demonstration purposes, we store the radial velocity, effective temperature, and G-band magnitude using a float32 for each field. The size of the table is 14.1 GB uncompressed, but it can be compressed to 5.6 GB. Adding another 1.5 GB for the index brings the total size to 7.2 GB. Therefore, the 3D array is 6.5 GB, and the table with the additional information and its index are 7.2 GB, making a total of 13.7 GB. This comfortably fits within the storage capacity of any modern laptop.

.. figure:: 3d-view-milkyway-inverse.png
   :scale: 25%

   Gaia DR3 dataset as a 3D array (preliminary, this is not from the dataset in this paper). :label:`gaia-3d-dset`

Figure :ref:`gaia-3d-dset` shows a 3D view of the Milky Way different type of stars. Each point is a star, and the color of each point represents the star's magnitude, with the brightest stars appearing as the reddest points. Although this view provides a unique perspective, the dimensions of the cube are not enough to fully capture the spiral arms of the Milky Way.

One advantage of using a 3D array is the ability to utilize Blosc2 NDim's powerful slicing capabilities for quickly exploring parts of the dataset. For example, we could search for star clusters by extracting small cubes as NumPy arrays, and counting the number of stars in each one. A cube containing an abnormally high number of stars would be a candidate for a cluster. We could also extract a thin 3D slice of the cube and project it as a 2D image, where the pixels colors represent the magnitude of the shown stars. This could be used to generate a cinematic view of a journey over different trajectories in the Milky Way.

Blosc2 NDim
-----------

Blosc2 NDim is a new feature of Blosc2 that allows to create and read n-dimensional datasets in an extremely efficient way thanks to a completely general n-dim 2-level partitioning, allowing to slice and dice arbitrary large (and compressed!) data in a more fine-grained way. Having a second partition provides a better flexibility to fit the different partitions at the different CPU cache levels, making compression even more efficient.

.. figure:: b2nd-2level-parts.png
   :scale: 12%

   Blosc2 NDim 2-level partitioning. :label:`b2nd-2level-parts`

.. figure:: b2nd-3d-dset.png
   :scale: 40%

   Blosc2 NDim 2-level partitioning is flexible. The dimensions of both partitions can be specified in any arbitrary way that fits the expected read access patterns. :label:`b2nd-3d-dset`

With these finer-grained cubes (also known as partitions), arbitrary n-dimensional slices can be retrieved faster because not all the data necessary for the coarser-grained partition has to be decompressed, as usually happens in other libraries. See Figures :ref:`b2nd-2level-parts` and :ref:`b2nd-3d-dset` to learn how this works and how to set it up. Also, see Figure :ref:`read-partial-slices` for a comparison against other libraries that use just a single partition (e.g., HDF5, Zarr).

.. figure:: read-partial-slices.png
   :scale: 70%

   Speed comparison for reading partial n-dimensional slices of a 4D dataset. The legends labeled "DIM N" refer to slices taken orthogonally to each dimension. The sizes for the two partitions have been chosen such that the first partition fits comfortably in the L3 cache of the CPU (Intel i9 13900K), and the second partition fits in the L1 cache of the CPU. :cite:`BDT23-blosc2-ndim-intro`. :label:`read-partial-slices`

It is important to note that Blosc2 NDim supports all data types in NumPy. This means that, in addition to the typical data types like signed/unsigned int, single and double-precision floats, bools or strings, it can also store datetimes (including units), and arbitrarily nested heterogeneous types. This allows to create multidimensional tables and more.

Support for multiple codecs, filters, and other compression features
---------------------------------------------------------------------

Blosc2 is not only a compression library, but also a framework for creating efficient compression pipelines. A compression pipeline is composed of a sequence of filters, followed by a compression codec. A filter is a transformation that is applied to the data before compression, and a codec is a compression algorithm that is applied to the filtered data. Filters can lead to better compression ratios and improved compression/decompression speeds.

Blosc2 supports a variety of codecs, filters, and other compression features. In particular, it supports the following codecs out-of-the-box:

- BloscLZ (fast codec, the default),
- LZ4 (a very fast codec),
- LZ4HC (high compression variant of LZ4),
- Zlib (the Zlib-NG variant of Zlib),
- Zstd (high compression), and
- ZFP (lossy compression for n-dimensional datasets of floats).

It also supports the following filters out-of-the-box:

- Shuffle (groups equal significant bytes together, useful for ints/floats),
- Shuffle with bytedelta (same than shuffle, but storing deltas of consecutive same significant bytes),
- Bitshuffle (groups equal significant bits together, useful for ints/floats), and
- Truncation (truncates precision, useful for floats; lossy).

Blosc2 utilizes a pipeline architecture that enables the chaining of different filters :cite:`BDT22-blosc2-pipeline` followed by a compression codec. Additionally, it allows for pre-filters (user code meant to be executed before the pipeline) and post-filters (user code meant to be executed after the pipeline). This architecture is highly flexible and minimizes data copies between the different steps, making it possible to create highly efficient pipelines for a variety of use cases. Figure :ref:`blosc2-pipeline` illustrates how this works.

.. figure:: blosc2-pipeline-v2.png

   The Blosc2 filter pipeline. During compression, the first function applied is the prefilter (if any), followed by the filter pipeline (with a maximum of six filters), and finally, the codec. During decompression, the order is reversed: first the codec, then the filter pipeline, and finally the postfilter (if any). :label:`blosc2-pipeline`

Furthermore, Blosc2 supports user-defined codecs and filters, allowing one to create their own compression algorithms and use them within Blosc2 :cite:`BDT22-blosc2-pipeline`. These user-defined codecs and filters can also be dynamically loaded :cite:`BDT23-dynamic-plugins`, registered globally within Blosc2, and installed via a Python wheel so that they can be used seamlessly from any Blosc2 application (whether in C, Python, or any other language that provides a Blosc2 wrapper).

Automatic tuning of compression parameters
------------------------------------------

Finding the right compression parameters for the data is probably the most difficult part of using a compression library. Which combination of code and filter would provide the best compression ratio? Which one would provide the best compression/decompression speed?

BTune is an AI tool for Blosc2 that automatically finds the optimal combination of compression parameters to suit the user needs. It uses a neural network that is trained on the most representative datasets to be compressed. This allows it to predict the best compression parameters based on a given balance between compression ratio and compression/decompression speed.

For example, Table :ref:`predicted-dparams-example` displays the results for the predicted compression parameters tuned for decompression speed. Curiously, fast decompression does not necessarily imply fast compression. This table is provided to the user so that he/she can choose the best balance value for his/her needs.

.. table:: BTune prediction of the best compression parameters for decompression speed, depending on a balance value between compression ratio and decompression speed (0 means favoring speed only, and 1 means favoring compression ratio only). It can be seen that BloscLZ + Shuffle is the most predicted category when decompression speed is preferred, whereas Zstd + Shuffle + ByteDelta is the most predicted one when the specified balance is towards optimizing for the compression ratio.  Speeds are in GB/s.  :label:`predicted-dparams-example`

   +---------+-------------------+---------+--------+--------+
   | Balance | Most predicted    |  Cratio | Cspeed | Dspeed |
   +=========+===================+=========+========+========+
   | 0.0     | blosclz-shuffle-5 | 2.09    | 14.47  | 48.93  |
   +---------+-------------------+---------+--------+--------+
   | 0.1     | blosclz-shuffle-5 | 2.09    | 14.47  | 48.93  |
   +---------+-------------------+---------+--------+--------+
   | 0.2     | blosclz-shuffle-5 | 2.09    | 14.47  | 48.93  |
   +---------+-------------------+---------+--------+--------+
   | 0.3     | blosclz-shuffle-5 | 2.09    | 14.47  | 48.93  |
   +---------+-------------------+---------+--------+--------+
   | 0.4     | zstd-bytedelta-1  | 3.30    | 17.04  | 21.65  |
   +---------+-------------------+---------+--------+--------+
   | 0.5     | zstd-bytedelta-1  | 3.30    | 17.04  | 21.65  |
   +---------+-------------------+---------+--------+--------+
   | 0.6     | zstd-bytedelta-1  | 3.30    | 17.04  | 21.65  |
   +---------+-------------------+---------+--------+--------+
   | 0.7     | zstd-bytedelta-1  | 3.30    | 17.04  | 21.65  |
   +---------+-------------------+---------+--------+--------+
   | 0.8     | zstd-bytedelta-1  | 3.30    | 17.04  | 21.65  |
   +---------+-------------------+---------+--------+--------+
   | 0.9     | zstd-bytedelta-1  | 3.30    | 17.04  | 21.65  |
   +---------+-------------------+---------+--------+--------+
   | 1.0     | zstd-bytedelta-9  | 3.31    | 0.07   | 11.40  |
   +---------+-------------------+---------+--------+--------+

On the other hand, Table :ref:`predicted-cparams-example`, shows an example of predicted compression parameter tuned for compression speed and ratio on a different dataset.

.. table:: BTune prediction of the best compression parameters for compression speed, depending on a balanced value. It can be seen that LZ4 + Bitshuffle is the most predicted category when compression speed is preferred, whereas Zstd + Shuffle + ByteDelta is the most predicted one when the specified balance is leveraged towards the compression ratio. Speeds are in GB/s. :label:`predicted-cparams-example`

   +---------+------------------+---------+--------+--------+
   | Balance | Most predicted   |  Cratio | Cspeed | Dspeed |
   +=========+==================+=========+========+========+
   | 0.0     | lz4-bitshuffle-5 | 3.41    | 21.78  | 32.0   |
   +---------+------------------+---------+--------+--------+
   | 0.1     | lz4-bitshuffle-5 | 3.41    | 21.78  | 32.0   |
   +---------+------------------+---------+--------+--------+
   | 0.2     | lz4-bitshuffle-5 | 3.41    | 21.78  | 32.0   |
   +---------+------------------+---------+--------+--------+
   | 0.3     | lz4-bitshuffle-5 | 3.41    | 21.78  | 32.0   |
   +---------+------------------+---------+--------+--------+
   | 0.4     | lz4-bitshuffle-5 | 3.41    | 21.78  | 32.0   |
   +---------+------------------+---------+--------+--------+
   | 0.5     | lz4-bitshuffle-5 | 3.41    | 21.78  | 32.0   |
   +---------+------------------+---------+--------+--------+
   | 0.6     | lz4-bitshuffle-5 | 3.41    | 21.78  | 32.0   |
   +---------+------------------+---------+--------+--------+
   | 0.7     | lz4-bitshuffle-5 | 3.41    | 21.78  | 32.0   |
   +---------+------------------+---------+--------+--------+
   | 0.8     | zstd-bytedelta-1 | 3.98    | 9.41   | 18.8   |
   +---------+------------------+---------+--------+--------+
   | 0.9     | zstd-bytedelta-1 | 3.98    | 9.41   | 18.8   |
   +---------+------------------+---------+--------+--------+
   | 1.0     | zstd-bytedelta-9 | 4.06    | 0.15   | 14.1   |
   +---------+------------------+---------+--------+--------+

After training the neural network, the BTune plugin can automatically tune the compression parameters for a given dataset. During inference, the user can set the preferred balance by setting the :code:`BTUNE_BALANCE` environment variable to a floating point value between 0 and 1. A value of 0 favors speed only, while a value of 1 favors compression ratio only. This setting automatically selects the compression parameters most suitable to the current data whenever a new Blosc2 data container is created.

Ingesting and processing data of Gaia
-------------------------------------

The raw data of Gaia is stored in CSV files.  The coordinates are stored in the gaia_source directory (http://cdn.gea.esac.esa.int/Gaia/gdr3/gaia_source/).  These can be easily parsed and ingested as Blosc2 files with the following code:

.. code-block:: python

   def load_rawdata(out="gaia.b2nd"):
       dtype = {"ra": np.float32,
                "dec": np.float32,
                "parallax": np.float32}
       barr = None
       for file in glob.glob("gaia-source/*.csv*"):
           # Load raw data
           df = pd.read_csv(
               file,
               usecols=["ra", "dec", "parallax"],
               dtype=dtype, comment='#')
           # Convert to numpy array and remove NaNs
           arr = df.to_numpy()
           arr = arr[~np.isnan(arr[:, 2])]
           if barr is None:
               # Create a new Blosc2 file
               barr = blosc2.asarray(
                   arr,
                   chunks=(2**20, 3),
                   urlpath=out,
                   mode="w")
           else:
               # Append to existing Blosc2 file
               barr.resize(
                   (barr.shape[0] + arr.shape[0], 3))
               barr[-arr.shape[0]:] = arr
       return barr

Once we have the raw data in a Blosc2 container, we can select the stars in a radius of 10 thousand light years using this function:

.. code-block:: python

   def convert_select_data(fin="gaia.b2nd",
                           fout="gaia-ly.b2nd"):
       barr = blosc2.open(fin)
       ra = barr[:, 0]
       dec = barr[:, 1]
       parallax = barr[:, 2]
       # 1 parsec = 3.26 light years
       ly = ne.evaluate("3260 / parallax")
       # Remove ly < 0 and > 10_000
       valid_ly = ne.evaluate(
           "(ly > 0) & (ly < 10_000)")
       ra = ra[valid_ly]
       dec = dec[valid_ly]
       ly = ly[valid_ly]
       # Cartesian x, y, z from spherical ra, dec, ly
       x = ne.evaluate("ly * cos(ra) * cos(dec)")
       y = ne.evaluate("ly * sin(ra) * cos(dec)")
       z = ne.evaluate("ly * sin(dec)")
       # Save to a new Blosc2 file
       out = blosc2.zeros(mode="w", shape=(3, len(x)),
                          dtype=x.dtype, urlpath=fout)
       out[0, :] = x
       out[1, :] = y
       out[2, :] = z
       return out


Finally, we can compute the density of stars in a 3D grid with this script:

.. code-block:: python

   R = 1  # resolution of the 3D cells in ly
   LY_RADIUS = 10_000  # radius of the sphere in ly
   CUBE_SIDE = (2 * LY_RADIUS) // R
   MAX_STARS = 1000_000_000  # max number of stars to load

   b = blosc2.open("gaia-ly.b2nd")
   x = b[0, :MAX_STARS]
   y = b[1, :MAX_STARS]
   z = b[2, :MAX_STARS]

   # Create 3d array.
   # Be sure to have enough swap memory (around 8 TB!)
   a3d = np.zeros((CUBE_SIDE, CUBE_SIDE, CUBE_SIDE),
                  dtype=np.float32)
   for i, coords in enumerate(zip(x, y, z)):
       x_, y_, z_ = coords
       a3d[(np.floor(x_) + LY_RADIUS) // R,
           (np.floor(y_) + LY_RADIUS) // R,
           (np.floor(z_) + LY_RADIUS) // R] += 1

   # Save 3d array as Blosc2 NDim file
   blosc2.asarray(a3d,
                  urlpath="gaia-3d.b2nd", mode="w",
                  chunks=(200, 200, 200),
                  blocks=(20, 20, 20),
                  )

With that, we have a 3D array of shape 20,000 x 20,000 x 20,000 with the number of stars with a 1 light year resolution.  We can visualize it with the following code:

To be completed ...

Conclusions
-----------

Working with large, multi-dimensional data cubes can be challenging due to the costly data handling involved. In this document, we demonstrate how the two-partition feature in Blosc2 NDim can help reduce the amount of data movement required when retrieving thin slices of large datasets. Additionally, this feature provides a foundation for leveraging cache hierarchies in modern CPUs.

Blosc2 supports a variety of compression codecs and filters, making it easier to select the most appropriate ones for the dataset being explored. It also supports storage in either memory or on disk, which is crucial for large datasets. Another important feature is the ability to store data in a container format that can be easily shared across different programming languages. Furthermore, Blosc2 has special support for sparse datasets, which greatly improves the compression ratio in this scenario.

We have also shown how the BTune plugin can be used to automatically tune the compression parameters for a given dataset.  This is especially useful when we want to compress data efficiently, but we do not know the best compression parameters beforehand.

In conclusion, we have shown how to utilize the Blosc2 library for storing and processing the Gaia dataset. This dataset serves as a prime example of a large, multi-dimensional dataset that can be efficiently stored and processed using Blosc2 NDim.
