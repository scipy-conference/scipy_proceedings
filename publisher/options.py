"""
Expose options from scipy_conf.cfg as a dictionary.
"""

__all__ = ['options']

from ConfigParser import ConfigParser
import os.path
import codecs

default_filename = os.path.join(os.path.dirname(__file__),
                                '../scipy_proc.cfg')

def cfg2dict(filename=default_filename):
    """Return the content of a .ini file as a dictionary.

    """
    options = {}

    if not os.path.isfile(filename):
        print "*** Warning: Could not load config file '%s'." % filename
    else:
        cp = ConfigParser()
        with codecs.open(filename, encoding='utf-8', mode='r') as fh:
            cp.readfp(fh)
            for key in cp.options('default'):
                options[key] = cp.get('default', key)

    return options

def dict2cfg(d, filename):
    """Write dictionary out to config file.

    """

    with codecs.open(filename, encoding='utf-8', mode='w') as f:
        f.write('[default]\n')
        for key, value in d.items():
            f.write('%s = %s\n' % (key, value))

options = cfg2dict()

