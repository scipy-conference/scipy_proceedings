"""
Utilities used by more than one module. 
"""

import glob 
import os

def glob_for_one_file(path, pattern):
    """
    This checks whether a single file matching the pattern is found in path.

    It uses glob's default matching, and returns a RuntimeError if the condition is not met.
    """

    try:
        file_found, = glob.glob(os.path.join(path, pattern))
    except ValueError:
        raise RuntimeError("Found more than one input matching {}--not sure which "
                           "one to use.".format(pattern))

    return file_found
