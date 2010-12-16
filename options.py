"""
Expose options from scipy_conf.cfg as a dictionary.
"""

__all__ = ['options']

from ConfigParser import ConfigParser

options = {}

cp = ConfigParser()
cp.read('scipy_proc.cfg')
for key in cp.options('default'):
    options[key] = cp.get('default', key)
