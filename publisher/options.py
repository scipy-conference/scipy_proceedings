"""
Expose options from scipy_conf.cfg as a dictionary.
"""

__all__ = ['options']

from ConfigParser import ConfigParser
import os.path

options = {}

cp = ConfigParser()
cp.read(os.path.join(os.path.dirname(__file__), '../scipy_proc.cfg'))
for key in cp.options('default'):
    options[key] = cp.get('default', key)
