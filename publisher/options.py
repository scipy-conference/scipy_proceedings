"""
Configuration utilities.
"""

__all__ = ['options']

import os.path
import json

default_filename = os.path.join(os.path.dirname(__file__),
                                '../scipy_proc.json')

def cfg2dict(filename=default_filename):
    """Return the content of a .ini file as a dictionary.

    """
    if not os.path.exists(filename):
        print '*** Warning: %s does not exist.' % filename
        return {}

    return json.load(open(filename, 'r'))

def dict2cfg(d, filename):
    """Write dictionary out to config file.

    """
    json.dump(d, open(filename, 'w'))

options = cfg2dict()
