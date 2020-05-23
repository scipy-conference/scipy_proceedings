:author: Jim Pivarski
:email: pivarski@princeton.edu
:institution: Princeton University

:author: Ianna Osborne
:email: Ianna.Osborne@cern.ch
:institution: Princeton University

:author: Pratyush Das
:email: reikdas@gmail.com
:institution: Institute of Engineering and Management

:author: Anish Biswas
:email: anishbiswas271@gmail.com
:institution: Manipal Institute of Technology

:author: Peter Elmer
:email: Peter.Elmer@cern.ch
:institution: Princeton University

------------------------------------------------
Awkward Array: JSON-like data, NumPy-like idioms
------------------------------------------------

.. class:: abstract

    NumPy simplifies and accelerates mathematical calculations in Python, but only for rectilinear arrays of numbers. Awkward Array provides a similar interface for JSON-like data: slicing, masking, broadcasting, and performing vectorized math on the attributes of objects, unequal-length nested lists (i.e. ragged/jagged arrays), and heterogeneous data types. Awkward Arrays are columnar data structures, like (and convertible to/from) Apache Arrow, with a focus on manipulation, rather than serialization/transport. These arrays can be passed between C++ and Python, and they can be used in functions that are JIT-compiled by Numba.

.. class:: keywords

   NumPy, Numba, C++, Apache Arrow, Columnar data, AOS-to-SOA, Ragged array, Jagged array, JSON

Motivation from particle physics
--------------------------------

asdf

Demonstration with GeoJSON bike routes
--------------------------------------

asdf

Data types and common operations
--------------------------------

types: lists, records, missing data, heterogeneous, virtual, partitioned

operations: slice/mask with variable-width/missing data extensions, reducers, num, zip/unzip, flatten, pad_none/fill_none, cartesian, combinations

Columnar representation, columnar implementation
------------------------------------------------

asdf (talk about Arrow in this section, maybe also Pandas)

Numba for just-in-time compilation
----------------------------------

asdf

ArrayBuilder: creating columnar data in-place
---------------------------------------------

asdf

High-level behaviors
--------------------

asdf

GPU backend
-----------

asdf

Conclusions
-----------

asdf

References
----------

asdf
